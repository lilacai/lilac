<script lang="ts">
  import type {SvelteComponent} from 'svelte';

  export let text: string | undefined;
  export let x: number;
  export let y: number;
  export let component: typeof SvelteComponent | undefined;
  export let props: Record<string, unknown> | undefined;
  const pageWidth = window.innerWidth;
  const marginPx = 10;

  let width = 0;
</script>

{#if text != ''}
  <div
    role="tooltip"
    class="hover-tooltip absolute mt-2
    -translate-x-1/2 break-words border border-gray-300 bg-white p-2 shadow-md"
    style:top="{y}px"
    style:left="{Math.max(width / 2 + marginPx, Math.min(x, pageWidth - width / 2 - marginPx))}px"
    bind:clientWidth={width}
  >
    <div class="min-w-xl text-container">
      {#if text}
        <span class="whitespace-pre-wrap">{text}</span>
      {:else if component}
        <svelte:component this={component} {...props} />
      {/if}
    </div>
  </div>
{/if}

<style>
  .text-container {
    min-width: 4rem;
    max-width: fit-content;
  }
</style>
