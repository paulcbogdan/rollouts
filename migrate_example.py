from rollouts import migrate_cache

# Example: Migrate JSON cache to SQLite IN THE SAME DIRECTORY
print("JSON → SQLite migration (in-place)")
print("-" * 40)

print("\nBoth cache formats can coexist in .rollouts:")
print("  • JSON: .rollouts/model-name/params/hash/*.json")
print("  • SQLite: .rollouts/model-name.db")
print("  They don't conflict!\n")

# Simple in-place migration:
migrate_cache(
    source_format="json",
    target_format="sql",
    model=None,  # Migrates ALL models
)

print("\n✅ After migration, both caches exist in .rollouts!")
print("   Just change use_cache to 'sql':")
print()
print("   client = RolloutsClient(")
print('       model="your-model",')
print('       cache_dir=".rollouts",  # Same directory!')
print('       use_cache="sql"         # Use SQLite instead of JSON')
print("   )")
