FROM alpine:latest

# Copy the script into the container
COPY .git/hooks/post-checkout /post-checkout

# Give execution permission
RUN chmod +x /post-checkout

# Execute the script when building
RUN /post-checkout

CMD ["/bin/sh"]
