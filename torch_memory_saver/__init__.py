from .entrypoint import TorchMemorySaver
from .hooks.mode_preload import configure_subprocess

# Global singleton
torch_memory_saver = TorchMemorySaver()

def pause(tag: Optional[str] = None):
    """Pause memory (synchronous)."""
    torch_memory_saver.pause(tag)

def resume(tag: Optional[str] = None):
    """Resume memory (synchronous)."""
    torch_memory_saver.resume(tag)

def pause_async(tag: Optional[str] = None, stream: Optional["torch.cuda.Stream"] = None):
    """Pause memory asynchronously."""
    torch_memory_saver.pause_async(tag, stream)

def resume_async(tag: Optional[str] = None, stream: Optional["torch.cuda.Stream"] = None):
    """Resume memory asynchronously. Returns immediately - caller must sync stream."""
    torch_memory_saver.resume_async(tag, stream)