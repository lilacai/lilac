FROM alpine:latest

# Copy the script to the container
COPY entrypoint.sh /entrypoint.sh

# Give execute permissions
RUN chmod +x /entrypoint.sh

# Run the script when the container builds
RUN /entrypoint.sh

CMD ["/bin/sh"]
