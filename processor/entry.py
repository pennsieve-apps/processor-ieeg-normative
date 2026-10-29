"""
One image, two runtimes: as a Lambda function (AWS_LAMBDA_RUNTIME_API is set) it hands over
to the Lambda runtime interface client with processor.handler; as a container it runs main()
once.
"""
import os
import sys

if os.environ.get("AWS_LAMBDA_RUNTIME_API"):
    os.execvp(sys.executable, [sys.executable, "-m", "awslambdaric", "processor.handler.handler"])

from processor.main import main  # noqa: E402

sys.exit(main())
