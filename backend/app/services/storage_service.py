"""
S3-compatible storage service for file management.

Supports AWS S3, Wasabi, MinIO, and other S3-compatible services.
Files are organized by tenant for complete isolation.
"""
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO, AsyncGenerator
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import mimetypes
import hashlib
import logging
import asyncio
from functools import partial
from io import BytesIO

from app.core.config import settings
from app.core.tenant import get_current_tenant, TenantContext

logger = logging.getLogger(__name__)


class StorageService:
    """
    S3-compatible storage service for file operations.

    All operations are tenant-aware - files are stored under
    tenant-specific prefixes for isolation.
    """

    def __init__(self):
        """Initialize S3 client with configuration."""
        self._client = None
        self._resource = None

    @property
    def client(self):
        """Lazy-load S3 client."""
        if self._client is None:
            self._client = boto3.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT,
                region_name=settings.S3_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                config=Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'}  # Use path-style for compatibility
                )
            )
        return self._client

    @property
    def resource(self):
        """Lazy-load S3 resource for higher-level operations."""
        if self._resource is None:
            self._resource = boto3.resource(
                's3',
                endpoint_url=settings.S3_ENDPOINT,
                region_name=settings.S3_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                config=Config(
                    signature_version='s3v4',
                    s3={'addressing_style': 'path'}
                )
            )
        return self._resource

    @property
    def bucket_name(self) -> str:
        """Get configured bucket name."""
        return settings.S3_BUCKET

    def _get_tenant_prefix(self, tenant: Optional[TenantContext] = None) -> str:
        """
        Get the S3 key prefix for the current tenant.

        Files are organized as: kahflane/{tenant_slug}/...
        """
        if tenant is None:
            tenant = get_current_tenant()

        if not tenant:
            raise ValueError("No tenant context available for storage operations")

        return f"kahflane/{tenant.tenant_slug}"

    def _generate_file_key(
        self,
        filename: str,
        folder: str = "documents",
        tenant: Optional[TenantContext] = None,
    ) -> str:
        """
        Generate a unique S3 key for a file.

        Format: kahflane/{tenant_slug}/{folder}/{date}/{uuid}_{filename}

        Args:
            filename: Original filename
            folder: Folder within tenant (documents, avatars, etc.)
            tenant: Optional tenant context

        Returns:
            Full S3 key
        """
        prefix = self._get_tenant_prefix(tenant)
        date_path = datetime.utcnow().strftime("%Y/%m/%d")
        unique_id = uuid4().hex[:12]

        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ".-_")

        return f"{prefix}/{folder}/{date_path}/{unique_id}_{safe_filename}"

    def _get_content_type(self, filename: str) -> str:
        """Guess content type from filename."""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or 'application/octet-stream'

    async def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        folder: str = "documents",
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
        tenant: Optional[TenantContext] = None,
    ) -> dict:
        """
        Upload a file to S3.

        Args:
            file_data: File-like object or bytes
            filename: Original filename
            folder: Folder category (documents, avatars, etc.)
            content_type: MIME type (auto-detected if not provided)
            metadata: Additional metadata to store with file
            tenant: Optional tenant context

        Returns:
            Dict with file_key, file_url, file_size, content_type
        """
        key = self._generate_file_key(filename, folder, tenant)

        if content_type is None:
            content_type = self._get_content_type(filename)

        # Read file data
        if hasattr(file_data, 'read'):
            data = file_data.read()
        else:
            data = file_data

        file_size = len(data)

        # Calculate MD5 for integrity
        md5_hash = hashlib.md5(data).hexdigest()

        # Prepare metadata
        s3_metadata = {
            'original-filename': filename,
            'md5-hash': md5_hash,
        }
        if metadata:
            s3_metadata.update({k: str(v) for k, v in metadata.items()})

        # Upload to S3 (run in thread pool for async)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self.client.put_object,
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=s3_metadata,
            )
        )

        logger.info(f"Uploaded file to S3: {key} ({file_size} bytes)")

        return {
            'file_key': key,
            'file_url': f"s3://{self.bucket_name}/{key}",
            'file_size': file_size,
            'content_type': content_type,
            'md5_hash': md5_hash,
        }

    async def upload_bytes(
        self,
        data: bytes,
        filename: str,
        folder: str = "documents",
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
        tenant: Optional[TenantContext] = None,
    ) -> dict:
        """
        Upload bytes directly to S3.

        Convenience wrapper around upload_file for raw bytes.
        """
        return await self.upload_file(
            file_data=BytesIO(data),
            filename=filename,
            folder=folder,
            content_type=content_type,
            metadata=metadata,
            tenant=tenant,
        )

    async def download_file(
        self,
        file_key: str,
        tenant: Optional[TenantContext] = None,
    ) -> bytes:
        """
        Download a file from S3.

        Args:
            file_key: S3 key of the file
            tenant: Optional tenant context for validation

        Returns:
            File contents as bytes

        Raises:
            ValueError: If file doesn't belong to tenant
            FileNotFoundError: If file doesn't exist
        """
        # Validate tenant ownership
        if tenant is None:
            tenant = get_current_tenant()

        if tenant:
            expected_prefix = self._get_tenant_prefix(tenant)
            if not file_key.startswith(expected_prefix):
                raise ValueError("File does not belong to current tenant")

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self.client.get_object,
                    Bucket=self.bucket_name,
                    Key=file_key,
                )
            )
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File not found: {file_key}")
            raise

    async def download_file_stream(
        self,
        file_key: str,
        chunk_size: int = 8192,
        tenant: Optional[TenantContext] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Download a file as a stream for large files.

        Args:
            file_key: S3 key of the file
            chunk_size: Size of each chunk in bytes
            tenant: Optional tenant context for validation

        Yields:
            Chunks of file data
        """
        # Validate tenant ownership
        if tenant is None:
            tenant = get_current_tenant()

        if tenant:
            expected_prefix = self._get_tenant_prefix(tenant)
            if not file_key.startswith(expected_prefix):
                raise ValueError("File does not belong to current tenant")

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self.client.get_object,
                    Bucket=self.bucket_name,
                    Key=file_key,
                )
            )

            body = response['Body']
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File not found: {file_key}")
            raise

    async def delete_file(
        self,
        file_key: str,
        tenant: Optional[TenantContext] = None,
    ) -> bool:
        """
        Delete a file from S3.

        Args:
            file_key: S3 key of the file
            tenant: Optional tenant context for validation

        Returns:
            True if deleted successfully
        """
        # Validate tenant ownership
        if tenant is None:
            tenant = get_current_tenant()

        if tenant:
            expected_prefix = self._get_tenant_prefix(tenant)
            if not file_key.startswith(expected_prefix):
                raise ValueError("File does not belong to current tenant")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(
                    self.client.delete_object,
                    Bucket=self.bucket_name,
                    Key=file_key,
                )
            )
            logger.info(f"Deleted file from S3: {file_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete file {file_key}: {e}")
            return False

    async def delete_files(
        self,
        file_keys: list[str],
        tenant: Optional[TenantContext] = None,
    ) -> dict:
        """
        Delete multiple files from S3.

        Args:
            file_keys: List of S3 keys to delete
            tenant: Optional tenant context for validation

        Returns:
            Dict with deleted and failed lists
        """
        # Validate tenant ownership for all files
        if tenant is None:
            tenant = get_current_tenant()

        if tenant:
            expected_prefix = self._get_tenant_prefix(tenant)
            for key in file_keys:
                if not key.startswith(expected_prefix):
                    raise ValueError(f"File does not belong to current tenant: {key}")

        if not file_keys:
            return {'deleted': [], 'failed': []}

        objects = [{'Key': key} for key in file_keys]

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self.client.delete_objects,
                    Bucket=self.bucket_name,
                    Delete={'Objects': objects},
                )
            )

            deleted = [obj['Key'] for obj in response.get('Deleted', [])]
            failed = [obj['Key'] for obj in response.get('Errors', [])]

            logger.info(f"Deleted {len(deleted)} files from S3")

            return {'deleted': deleted, 'failed': failed}

        except ClientError as e:
            logger.error(f"Failed to delete files: {e}")
            return {'deleted': [], 'failed': file_keys}

    async def get_file_info(
        self,
        file_key: str,
        tenant: Optional[TenantContext] = None,
    ) -> Optional[dict]:
        """
        Get metadata for a file.

        Args:
            file_key: S3 key of the file
            tenant: Optional tenant context for validation

        Returns:
            Dict with file metadata or None if not found
        """
        # Validate tenant ownership
        if tenant is None:
            tenant = get_current_tenant()

        if tenant:
            expected_prefix = self._get_tenant_prefix(tenant)
            if not file_key.startswith(expected_prefix):
                raise ValueError("File does not belong to current tenant")

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self.client.head_object,
                    Bucket=self.bucket_name,
                    Key=file_key,
                )
            )

            return {
                'file_key': file_key,
                'size': response['ContentLength'],
                'content_type': response['ContentType'],
                'last_modified': response['LastModified'],
                'metadata': response.get('Metadata', {}),
                'etag': response.get('ETag', '').strip('"'),
            }

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise

    async def file_exists(
        self,
        file_key: str,
        tenant: Optional[TenantContext] = None,
    ) -> bool:
        """Check if a file exists in S3."""
        info = await self.get_file_info(file_key, tenant)
        return info is not None

    async def generate_presigned_url(
        self,
        file_key: str,
        expires_in: int = 3600,
        method: str = 'get_object',
        tenant: Optional[TenantContext] = None,
    ) -> str:
        """
        Generate a presigned URL for temporary access.

        Args:
            file_key: S3 key of the file
            expires_in: URL expiration in seconds (default 1 hour)
            method: S3 method ('get_object' for download, 'put_object' for upload)
            tenant: Optional tenant context for validation

        Returns:
            Presigned URL string
        """
        # Validate tenant ownership
        if tenant is None:
            tenant = get_current_tenant()

        if tenant:
            expected_prefix = self._get_tenant_prefix(tenant)
            if not file_key.startswith(expected_prefix):
                raise ValueError("File does not belong to current tenant")

        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            partial(
                self.client.generate_presigned_url,
                ClientMethod=method,
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_key,
                },
                ExpiresIn=expires_in,
            )
        )

        return url

    async def generate_upload_url(
        self,
        filename: str,
        folder: str = "documents",
        content_type: Optional[str] = None,
        expires_in: int = 3600,
        tenant: Optional[TenantContext] = None,
    ) -> dict:
        """
        Generate a presigned URL for direct client upload.

        Args:
            filename: Intended filename
            folder: Folder category
            content_type: Expected content type
            expires_in: URL expiration in seconds
            tenant: Optional tenant context

        Returns:
            Dict with upload_url, file_key, and fields
        """
        key = self._generate_file_key(filename, folder, tenant)

        if content_type is None:
            content_type = self._get_content_type(filename)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                self.client.generate_presigned_post,
                Bucket=self.bucket_name,
                Key=key,
                Fields={
                    'Content-Type': content_type,
                },
                Conditions=[
                    {'Content-Type': content_type},
                    ['content-length-range', 1, 100 * 1024 * 1024],  # Max 100MB
                ],
                ExpiresIn=expires_in,
            )
        )

        return {
            'upload_url': response['url'],
            'file_key': key,
            'fields': response['fields'],
            'content_type': content_type,
        }

    async def list_files(
        self,
        prefix: Optional[str] = None,
        folder: str = "documents",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
        tenant: Optional[TenantContext] = None,
    ) -> dict:
        """
        List files in a tenant's folder.

        Args:
            prefix: Additional prefix filter within folder
            folder: Folder category
            max_keys: Maximum number of files to return
            continuation_token: Token for pagination
            tenant: Optional tenant context

        Returns:
            Dict with files list and pagination info
        """
        tenant_prefix = self._get_tenant_prefix(tenant)
        full_prefix = f"{tenant_prefix}/{folder}/"

        if prefix:
            full_prefix = f"{full_prefix}{prefix}"

        params = {
            'Bucket': self.bucket_name,
            'Prefix': full_prefix,
            'MaxKeys': max_keys,
        }

        if continuation_token:
            params['ContinuationToken'] = continuation_token

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(self.client.list_objects_v2, **params)
        )

        files = []
        for obj in response.get('Contents', []):
            files.append({
                'file_key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'],
                'etag': obj.get('ETag', '').strip('"'),
            })

        return {
            'files': files,
            'is_truncated': response.get('IsTruncated', False),
            'next_token': response.get('NextContinuationToken'),
            'count': len(files),
        }

    async def copy_file(
        self,
        source_key: str,
        dest_filename: str,
        dest_folder: str = "documents",
        tenant: Optional[TenantContext] = None,
    ) -> dict:
        """
        Copy a file within the same tenant.

        Args:
            source_key: Source file S3 key
            dest_filename: Destination filename
            dest_folder: Destination folder
            tenant: Optional tenant context

        Returns:
            Dict with new file key and info
        """
        # Validate source belongs to tenant
        if tenant is None:
            tenant = get_current_tenant()

        if tenant:
            expected_prefix = self._get_tenant_prefix(tenant)
            if not source_key.startswith(expected_prefix):
                raise ValueError("Source file does not belong to current tenant")

        dest_key = self._generate_file_key(dest_filename, dest_folder, tenant)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(
                self.client.copy_object,
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': source_key},
                Key=dest_key,
            )
        )

        logger.info(f"Copied file from {source_key} to {dest_key}")

        return {
            'file_key': dest_key,
            'source_key': source_key,
        }


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service


# Backward compatibility alias
storage_service = get_storage_service()


# Convenience functions

async def upload_file(
    file_data: BinaryIO,
    filename: str,
    folder: str = "documents",
    **kwargs
) -> dict:
    """Upload a file to storage."""
    return await storage_service.upload_file(file_data, filename, folder, **kwargs)


async def download_file(file_key: str, **kwargs) -> bytes:
    """Download a file from storage."""
    return await storage_service.download_file(file_key, **kwargs)


async def delete_file(file_key: str, **kwargs) -> bool:
    """Delete a file from storage."""
    return await storage_service.delete_file(file_key, **kwargs)


async def get_presigned_url(file_key: str, expires_in: int = 3600, **kwargs) -> str:
    """Generate a presigned download URL."""
    return await storage_service.generate_presigned_url(file_key, expires_in, **kwargs)
