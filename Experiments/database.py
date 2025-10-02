from supabase import create_client

SUPABASE_URL = "https://btzmsixtmoqiiebkkecm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0em1zaXh0bW9xaWllYmtrZWNtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg5ODIzNjIsImV4cCI6MjA3NDU1ODM2Mn0.rhml48N6fO8V_Wsu7JI5dva3nPYjBMrnFuDjNsc-9zQ"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
