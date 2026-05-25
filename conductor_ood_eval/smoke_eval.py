import argparse
import os
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL_NAME", "smoke-model"))
    parser.add_argument("--project", default="conductor_smoke")
    parser.add_argument("--group", default="smoke_eval")
    parser.add_argument("--run-name", default="openai-compat-smoke")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Answer with one short sentence: what is 2 + 2?",
            }
        ],
        "temperature": 0.1,
        "max_tokens": 128,
    }

    start = time.monotonic()
    response_text = ""
    status_code = None
    error = None
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(f"{args.base_url}/chat/completions", json=payload)
            status_code = response.status_code
            response.raise_for_status()
            data = response.json()
            response_text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        error = str(exc)
    latency_s = time.monotonic() - start

    success = bool(response_text.strip()) and error is None
    print(f"success={success}")
    print(f"status_code={status_code}")
    print(f"latency_s={latency_s:.3f}")
    print(f"response_chars={len(response_text)}")
    if response_text:
        print(response_text[:500])
    if error:
        print(f"error={error}")

    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.project,
            group=args.group,
            name=args.run_name,
            config={
                "base_url": args.base_url,
                "model": args.model,
                "status_code": status_code,
            },
        )
        wandb.log(
            {
                "smoke/success": int(success),
                "smoke/latency_s": latency_s,
                "smoke/response_chars": len(response_text),
                "smoke/status_code": status_code or 0,
            }
        )
        run.finish()

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
