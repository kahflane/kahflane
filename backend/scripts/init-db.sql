-- Kahflane Database Initialization Script
-- Creates public schema tables for multi-tenant authentication

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Ensure we're in public schema
SET search_path TO public;

-- Tenant Registry Table
CREATE TABLE IF NOT EXISTS public.tenant (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slug VARCHAR(63) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    schema_name VARCHAR(63) UNIQUE NOT NULL,
    plan_type VARCHAR(50) NOT NULL DEFAULT 'free',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tenant_slug ON public.tenant(slug);
CREATE INDEX IF NOT EXISTS idx_tenant_domain ON public.tenant(domain) WHERE domain IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tenant_active ON public.tenant(is_active) WHERE is_active = TRUE;

-- User Authentication Table
CREATE TABLE IF NOT EXISTS public."user" (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_email ON public."user"(email);
CREATE INDEX IF NOT EXISTS idx_user_active ON public."user"(is_active) WHERE is_active = TRUE;

-- Tenant Membership Table
CREATE TABLE IF NOT EXISTS public.tenant_membership (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public."user"(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES public.tenant(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    invited_by_id UUID REFERENCES public."user"(id) ON DELETE SET NULL,
    joined_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_membership_user ON public.tenant_membership(user_id);
CREATE INDEX IF NOT EXISTS idx_membership_tenant ON public.tenant_membership(tenant_id);
CREATE INDEX IF NOT EXISTS idx_membership_default ON public.tenant_membership(user_id, is_default) WHERE is_default = TRUE;

-- Tenant Invitation Table
CREATE TABLE IF NOT EXISTS public.tenant_invitation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES public.tenant(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    token VARCHAR(255) UNIQUE NOT NULL,
    invited_by_id UUID NOT NULL REFERENCES public."user"(id) ON DELETE CASCADE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    accepted_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_invitation_token ON public.tenant_invitation(token);
CREATE INDEX IF NOT EXISTS idx_invitation_email ON public.tenant_invitation(email);
CREATE INDEX IF NOT EXISTS idx_invitation_tenant ON public.tenant_invitation(tenant_id);
CREATE INDEX IF NOT EXISTS idx_invitation_pending ON public.tenant_invitation(expires_at)
    WHERE accepted_at IS NULL AND expires_at > NOW();

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_tenant_updated_at ON public.tenant;
CREATE TRIGGER update_tenant_updated_at
    BEFORE UPDATE ON public.tenant
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_user_updated_at ON public."user";
CREATE TRIGGER update_user_updated_at
    BEFORE UPDATE ON public."user"
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (for application user if different from owner)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO kahflane;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO kahflane;

COMMENT ON TABLE public.tenant IS 'Tenant registry - each tenant gets their own schema';
COMMENT ON TABLE public."user" IS 'User authentication - shared across all tenants';
COMMENT ON TABLE public.tenant_membership IS 'User-Tenant access mapping';
COMMENT ON TABLE public.tenant_invitation IS 'Pending invitations to join a tenant';

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Kahflane public schema initialized successfully';
END $$;
