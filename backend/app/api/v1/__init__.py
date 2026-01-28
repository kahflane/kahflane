from fastapi import APIRouter

from app.api.v1 import auth, tenants, organizations, documents, search

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(tenants.router, prefix="/tenants", tags=["tenants"])
router.include_router(organizations.router, prefix="/organizations", tags=["organizations"])
router.include_router(documents.router, prefix="/documents", tags=["documents"])
router.include_router(search.router, prefix="/search", tags=["search"])
