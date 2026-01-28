"""
SMTP email service for sending transactional emails.

This service is synchronous and designed to run inside Dramatiq workers.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Synchronous SMTP email sender for use in Dramatiq workers."""

    def send_email(self, to: str, subject: str, html_body: str) -> None:
        """Send an email via SMTP."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.SMTP_FROM_EMAIL
        msg["To"] = to
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            if settings.SMTP_USE_TLS:
                server.starttls()
            if settings.SMTP_USER and settings.SMTP_PASSWORD:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM_EMAIL, to, msg.as_string())

        logger.info(f"Email sent to {to}: {subject}")

    def send_otp_email(self, to: str, otp: str) -> None:
        """Send email verification OTP."""
        self.send_email(
            to,
            f"Verify your email - {settings.APP_NAME}",
            f"<p>Your verification code is: <strong>{otp}</strong></p>"
            f"<p>This code expires in {settings.OTP_EXPIRE_MINUTES} minutes.</p>",
        )

    def send_password_reset_email(self, to: str, otp: str) -> None:
        """Send password reset OTP."""
        self.send_email(
            to,
            f"Reset your password - {settings.APP_NAME}",
            f"<p>Your password reset code is: <strong>{otp}</strong></p>"
            f"<p>This code expires in {settings.OTP_EXPIRE_MINUTES} minutes.</p>",
        )

    def send_invitation_email(
        self, to: str, tenant_name: str, invite_token: str, invited_by: str
    ) -> None:
        """Send tenant invitation email."""
        link = f"{settings.FRONTEND_BASE_URL}/invite/{invite_token}"
        self.send_email(
            to,
            f"You're invited to {tenant_name} - {settings.APP_NAME}",
            f"<p>{invited_by} has invited you to join <strong>{tenant_name}</strong>.</p>"
            f'<p><a href="{link}">Accept invitation</a></p>'
            f"<p>This invitation expires in {settings.INVITATION_EXPIRE_DAYS} days.</p>",
        )


def get_email_service() -> EmailService:
    return EmailService()
