import argparse
import os
import sys
import io
# Add current directory to path so modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.ingestion import CodebaseIngester
    from modules.chunking import CodeChunker
    from modules.embeddings import EmbeddingGenerator
    from modules.vector_store import VectorStore
    from modules.question_processor import QuestionProcessor
    from modules.rag_generator import RAGGenerator
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

try:

    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
            
    class Prompt:
        @staticmethod
        def ask(prompt):
            return input(prompt + " ")
            
    class Panel:
        @staticmethod
        def fit(content, **kwargs):
            return content
            
    class Markdown:
        def __init__(self, content):
            self.content = content
        def __str__(self):
            return self.content
            
    class Progress:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def add_task(self, *args, **kwargs):
            return None
        def update(self, *args, **kwargs):
            pass

console = Console()

class CLIAppState:
    def __init__(self):
        self.ingester = CodebaseIngester()
        self.chunker = CodeChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.question_processor = None
        self.rag_generator = None
        self.repository_loaded = False
        self.repository_path = None

state = CLIAppState()

def load_repo(repo_path):
    try:
        repo_path = os.path.abspath(repo_path)
        if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
            console.print(f"[bold red]Error:[/bold red] Invalid path: {repo_path}")
            return False

        state.vector_store.clear()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Ingesting files...", total=None)
            source_files = state.ingester.ingest(repo_path)
            
            if not source_files:
                console.print(f"[bold red]Error:[/bold red] No supported source files found in {repo_path}")
                return False

            progress.update(task, description="[cyan]Chunking files...")
            chunks = state.chunker.chunk_files(source_files)
            
            progress.update(task, description="[cyan]Generating embeddings...")
            embeddings = state.embedding_generator.embed_chunks(chunks)
            
            progress.update(task, description="[cyan]Storing in vector database...")
            state.vector_store.add_embeddings(embeddings, chunks)
            
            state.question_processor = QuestionProcessor(state.embedding_generator, state.vector_store)
            state.rag_generator = RAGGenerator(state.question_processor)
            
            state.repository_loaded = True
            state.repository_path = repo_path
            
        console.print(f"[bold green]Success![/bold green] Loaded repository at {repo_path}")
        console.print(f"Found [bold]{len(source_files)}[/bold] files, created [bold]{len(chunks)}[/bold] chunks.")
        return True

    except Exception as e:
        console.print(f"[bold red]Error loading repository:[/bold red] {e}")
        return False

def chat_loop():
    console.print(Panel.fit("Welcome to CodeChat CLI! Type 'exit' or 'quit' to stop.", title="CodeChat AI", style="bold blue"))
    
    while True:
        try:
            question = Prompt.ask("\n[bold green]You[/bold green]")
            if question.lower() in ('exit', 'quit'):
                break
            
            if not question.strip():
                continue
                
            if not state.repository_loaded:
                 console.print("[bold yellow]Warning:[/bold yellow] No repository loaded. Use --path to load one or reload.")
                 # Prompt for path if not loaded
                 path = Prompt.ask("[bold yellow]Enter repository path[/bold yellow]")
                 if load_repo(path):
                     continue
                 else:
                     continue

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console
            ) as progress:
                progress.add_task("[cyan]Thinking...", total=None)
                response = state.rag_generator.generate(question)
            
            console.print("\n[bold blue]CodeChat AI:[/bold blue]")
            console.print(Markdown(response.answer))
            
            if response.source_files:
                console.print("\n[dim]Sources:[/dim]")
                for src in response.source_files[:3]:
                     console.print(f"[dim]- {src}[/dim]")

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

def main():
    parser = argparse.ArgumentParser(description="CodeChat AI CLI")
    parser.add_argument("--start", action="store_true", help="Start the interactive chat mode")
    parser.add_argument("--path", type=str, help="Path to the repository to load immediately")
    
    args = parser.parse_args()
    
    if args.start:
        if args.path:
            load_repo(args.path)
        else:
             # Try to load current directory by default if valid
             pass 
        chat_loop()
    else:
        # If no args, just show help or start anyway? 
        # The user requested `codechat --start`, but `codechat` alone should probably also work or show help.
        # Let's default to starting chat if just `codechat` is called, but respect the request format.
        # If no arguments are provided, maybe print help.
        if len(sys.argv) == 1:
             parser.print_help()
             return

if __name__ == "__main__":
    main()
