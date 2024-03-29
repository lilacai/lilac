<script lang="ts">
  import {goto} from '$app/navigation';
  import {
    editConceptMutation,
    queryConceptModels,
    queryConcepts
  } from '$lib/queries/conceptQueries';
  import {queryAuthInfo} from '$lib/queries/serverQueries';
  import {queryEmbeddings} from '$lib/queries/signalQueries';
  import {createDatasetViewStore} from '$lib/stores/datasetViewStore';
  import {getNavigationContext} from '$lib/stores/navigationStore';
  import {datasetLink} from '$lib/utils';
  import type {Concept} from '$lilac';
  import {Button, ToastNotification} from 'carbon-components-svelte';
  import {View, ViewOff} from 'carbon-icons-svelte';
  import ThumbsDownFilled from 'carbon-icons-svelte/lib/ThumbsDownFilled.svelte';
  import ThumbsUpFilled from 'carbon-icons-svelte/lib/ThumbsUpFilled.svelte';
  import {get} from 'svelte/store';
  import Expandable from '../Expandable.svelte';
  import {hoverTooltip} from '../common/HoverTooltip';
  import ConceptExampleList from './ConceptExampleList.svelte';
  import ConceptMetrics from './ConceptMetrics.svelte';
  import ConceptPreview from './ConceptPreview.svelte';
  import DatasetFieldEmbeddingSelector from './DatasetFieldEmbeddingSelector.svelte';
  import ConceptLabeler from './labeler/ConceptLabeler.svelte';

  export let concept: Concept;

  const authInfo = queryAuthInfo();
  $: userId = $authInfo.data?.user?.id;

  const concepts = queryConcepts();
  const navState = getNavigationContext();

  $: conceptInfo = $concepts.data?.find(
    c => c.namespace === concept.namespace && c.name === concept.concept_name
  );
  $: canEditConcept = conceptInfo?.acls.write == true;

  const conceptMutation = editConceptMutation();
  const embeddings = queryEmbeddings();

  $: positiveExamples = Object.values(concept.data).filter(v => v.label == true);
  $: negativeExamples = Object.values(concept.data).filter(v => v.label == false);

  $: randomPositive = positiveExamples[Math.floor(Math.random() * positiveExamples.length)];

  // Apply to a dataset.
  let applyDataset: {namespace: string; name: string} | undefined | null = undefined;
  let applyPath: string[] | undefined;
  let applyEmbedding: string | undefined = undefined;
  function openDataset() {
    if (applyPath == null || applyEmbedding == null || applyDataset == null) {
      return;
    }
    const store = createDatasetViewStore(applyDataset.namespace, applyDataset.name);
    store.addSearch({
      path: applyPath,
      type: 'concept',
      concept_namespace: concept.namespace,
      concept_name: concept.concept_name,
      embedding: applyEmbedding
    });
    goto(datasetLink(applyDataset.namespace, applyDataset.name, $navState, get(store)));
  }

  function remove(id: string) {
    if (!concept.namespace || !concept.concept_name) return;
    $conceptMutation.mutate([concept.namespace, concept.concept_name, {remove: [id]}]);
  }

  function add(text: string, label: boolean) {
    if (!concept.namespace || !concept.concept_name) return;
    $conceptMutation.mutate([concept.namespace, concept.concept_name, {insert: [{text, label}]}]);
  }

  $: conceptModels = queryConceptModels(concept.namespace, concept.concept_name);
</script>

<div class="flex h-full w-full flex-col gap-y-8 px-10">
  <div>
    <div class="flex flex-row items-center text-2xl font-semibold">
      {concept.concept_name}
      {#if userId == concept.namespace && !concept.metadata?.is_public}
        <div
          use:hoverTooltip={{
            text: 'Your concepts are only visible to you when logged in with Google.'
          }}
        >
          <ViewOff class="ml-2" />
        </div>
      {:else if concept.metadata?.is_public}
        <div
          use:hoverTooltip={{
            text: 'This concept is publicly visible.'
          }}
        >
          <View class="ml-2" />
        </div>
      {/if}
    </div>
    {#if concept.metadata?.description}
      <div class="text text-base text-gray-600">{concept.metadata.description}</div>
    {/if}
  </div>

  <Expandable expanded>
    <div slot="above" class="text-md font-semibold">Try it</div>
    <div slot="below">
      {#if randomPositive}
        <ConceptPreview example={randomPositive} {concept} />
      {:else}
        <div class="text-gray-600">No examples to preview. Please add examples.</div>
      {/if}
    </div>
  </Expandable>

  <Expandable>
    <div slot="above" class="text-md font-semibold">Apply to a dataset</div>
    <div slot="below">
      <DatasetFieldEmbeddingSelector
        bind:dataset={applyDataset}
        bind:path={applyPath}
        bind:embedding={applyEmbedding}
      />
      {#if applyDataset != null && applyPath != null && applyEmbedding != null}
        <div class="mt-4">
          <Button iconDescription={'Open dataset and apply concept.'} on:click={() => openDataset()}
            >Search by concept
          </Button>
        </div>
      {:else}
        <ToastNotification
          hideCloseButton
          kind="warning"
          fullWidth
          lowContrast
          title="Choose a dataset with a computed embedding"
          caption={'Dataset has no fields with computed embeddings. ' +
            'Please compute an embedding index before you can search by concept.'}
        />
      {/if}
    </div>
  </Expandable>
  <Expandable>
    <div slot="above" class="text-md font-semibold">Collect labels</div>
    <div slot="below" class="w-full">
      {#if canEditConcept}
        <ConceptLabeler {concept} />
      {:else}
        <ToastNotification
          hideCloseButton
          kind="warning"
          fullWidth
          lowContrast
          title="You don't have permission to edit this concept"
          caption={'You can only edit concepts you created.'}
        />
      {/if}
    </div>
  </Expandable>
  {#if $embeddings.data}
    <Expandable expanded>
      <div slot="above" class="text-md font-semibold">Metrics</div>
      <div slot="below" class="model-metrics flex flex-wrap gap-x-4 gap-y-4">
        {#each $embeddings.data as embedding}
          {@const model = $conceptModels.data?.find(m => m.embedding_name == embedding.name)}
          <ConceptMetrics
            {concept}
            embedding={embedding.name}
            {model}
            isFetching={$conceptModels.isFetching}
          />
        {/each}
      </div>
    </Expandable>
  {/if}

  <div class="flex gap-x-4">
    <div class="flex w-0 flex-grow flex-col gap-y-4">
      <span class="flex items-center gap-x-2 text-lg"
        ><ThumbsUpFilled /> In concept ({positiveExamples.length} examples)</span
      >
      <ConceptExampleList
        data={positiveExamples}
        on:remove={ev => remove(ev.detail)}
        on:add={ev => add(ev.detail, true)}
        {canEditConcept}
      />
    </div>
    <div class="flex w-0 flex-grow flex-col gap-y-4">
      <span class="flex items-center gap-x-2 text-lg"
        ><ThumbsDownFilled />Not in concept ({negativeExamples.length} examples)</span
      >
      <ConceptExampleList
        data={negativeExamples}
        on:remove={ev => remove(ev.detail)}
        on:add={ev => add(ev.detail, false)}
        {canEditConcept}
      />
    </div>
  </div>
</div>
