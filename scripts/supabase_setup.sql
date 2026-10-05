-- Persistent storage for the deployed app's live AI layer (app/supabase_store.py).
-- Run once in the Supabase SQL editor. Holds only generated AI outputs (keyed by video ID,
-- pipeline version and model) and a daily generation counter: no user or visitor data.
-- Row Level Security is on with no policies, so only the server-side secret key, which
-- bypasses RLS, can read or write; the publishable key can't.

create table if not exists public.ai_cache (
  namespace text not null,
  key text not null,
  value jsonb not null,
  updated_at timestamptz not null default now(),
  primary key (namespace, key)
);

create table if not exists public.ai_quota (
  day date primary key,
  count integer not null default 0 check (count >= 0)
);

alter table public.ai_cache enable row level security;
alter table public.ai_quota enable row level security;
revoke all on public.ai_cache, public.ai_quota from anon, authenticated;

-- Takes one of today's (UTC) generation slots; false once daily_limit is reached. The
-- check and the increment are one statement, so concurrent sessions can't overshoot.
create or replace function public.reserve_generation(daily_limit integer)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  today date := (now() at time zone 'utc')::date;
  taken integer;
begin
  insert into ai_quota (day, count) values (today, 0) on conflict (day) do nothing;
  update ai_quota set count = count + 1
    where day = today and count < daily_limit
    returning count into taken;
  return taken is not null;
end;
$$;

revoke execute on function public.reserve_generation(integer) from public, anon, authenticated;
grant execute on function public.reserve_generation(integer) to service_role;
