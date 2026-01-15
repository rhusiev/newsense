FROM rust:alpine AS builder
WORKDIR /app
RUN apk add --no-cache musl-dev

COPY Cargo.toml Cargo.lock ./
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release --target x86_64-unknown-linux-musl && \
    rm -rf src

COPY . .
RUN touch src/main.rs
ENV SQLX_OFFLINE=true
RUN cargo build --release --target x86_64-unknown-linux-musl

FROM alpine:latest
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/newsense-aggregator /usr/local/bin/
CMD ["newsense-aggregator"]
