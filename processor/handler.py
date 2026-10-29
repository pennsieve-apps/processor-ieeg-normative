"""
The Lambda runtime. A lambda step's payload carries inputDir, outputDir and the step's
parameters as fields (where a container gets them as environment variables); the handler
sets those variables and runs the same main(), so both runtimes write the same tables.
"""
import json
import os

from processor import main as app

PARAMETERS = tuple(app.DEFAULTS)


def handler(event, context):
    os.environ["INPUT_DIR"] = event.get("inputDir") or ""
    os.environ["OUTPUT_DIR"] = event.get("outputDir") or ""
    # A warm function keeps its environment between invocations, so a parameter this run
    # doesn't set is cleared, not left over from the last one.
    for name in PARAMETERS:
        value = event.get(name)
        if value is None:
            os.environ.pop(name, None)
        elif isinstance(value, bool):
            os.environ[name] = str(value).lower()
        elif isinstance(value, (dict, list)):
            os.environ[name] = json.dumps(value)
        else:
            os.environ[name] = str(value)
    code = app.main()
    if code != 0:
        # An exception, so the step fails; a returned value would succeed.
        raise RuntimeError(f"{app.APP_NAME} exited with {code}")
    return {"status": "ok", "outputDir": os.environ["OUTPUT_DIR"]}
