# Inject the Windows OS certificate store into Python's ssl module so that
# corporate proxy CA certs are trusted — same approach git uses via schannel.
# This is a no-op if truststore is not installed.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
