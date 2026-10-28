"""Build FlagOS submission ZIPs: one archive per task containing [Kernel Name].py."""
import os, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name in ("softcap_row_argmax", "rmsnorm_fp32", "gemma_fused_add_rmsnorm"):
    dst = os.path.join(ROOT, "submissions", f"{name}.zip")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(ROOT, "kernels", f"{name}.py"), f"{name}.py")
    print(dst, os.path.getsize(dst))
