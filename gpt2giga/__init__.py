import sys

from gpt2giga.api_server import run

__all__ = ["run", "main"]


def main():
    """Main entry point that routes to subcommands or default server."""
    # Check if running as subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "index-codebase":
        from gpt2giga.tools.indexer import index_codebase_cli

        # Remove subcommand from argv and pass to index_codebase_cli
        original_argv = sys.argv[:]
        sys.argv = sys.argv[1:]
        sys.argv[0] = "gpt2giga-index-codebase"  # Update script name
        try:
            index_codebase_cli()
        finally:
            # Restore original argv (for testing/debugging)
            sys.argv = original_argv
    else:
        # Run default proxy server
        run()


# Support subcommands when run as module or script
if __name__ == "__main__":
    main()
else:
    # When imported, set run as main entry point for poetry script
    # But also export main for explicit subcommand handling
    pass
