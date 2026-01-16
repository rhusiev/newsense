#!/usr/bin/env python3

import sys
import subprocess

DOCKER_COMPOSE_FILES = ["-f", "docker-compose.yml", "-f", "docker-compose.standalone.yml"]


def run_docker_commands(version: str) -> None:
    env = {"VERSION": version}
    
    build_cmd = ["docker", "compose"] + DOCKER_COMPOSE_FILES + ["build"]
    push_cmd = ["docker", "compose"] + DOCKER_COMPOSE_FILES + ["push"]
    
    print(f"Building with VERSION={version}...")
    subprocess.run(build_cmd, env={**subprocess.os.environ, **env}, check=True)
    
    print(f"Pushing with VERSION={version}...")
    subprocess.run(push_cmd, env={**subprocess.os.environ, **env}, check=True)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    
    run_docker_commands(version)
    run_docker_commands("latest")
    
    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()
