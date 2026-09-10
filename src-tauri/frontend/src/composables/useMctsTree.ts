import { reactive } from 'vue';
import { api } from '../api/client';
import type { MctsEdge, MctsNodeDetail, MctsRootInfo } from '../api/types';
import { useToast } from './useToast';

const toast = useToast();

export const MCTS_MAX_CHILDREN = 12;

export interface MctsLayoutNode {
  id: number;
  depth: number;
  edge: MctsEdge | null;
  n: number;
  q: number;
  children: MctsLayoutNode[];
  hiddenCount: number;
  x: number;
  y: number;
}

interface MctsStore {
  rootInfo: MctsRootInfo | null;
  expanded: Set<number>;
  showAll: boolean;
  searching: boolean;
}

const store = reactive<MctsStore>({
  rootInfo: null,
  expanded: new Set(),
  showAll: false,
  searching: false,
});

const childrenCache = new Map<number, MctsEdge[]>();
const summaries = new Map<number, { n: number; q: number }>();

export const treeTransform = reactive({ scale: 1, tx: 0, ty: 0 });

function edgeSummary(edge: MctsEdge): { n: number; q: number } {
  return { n: edge.n, q: edge.q };
}

async function ensureChildren(nodeId: number): Promise<MctsEdge[]> {
  const cached = childrenCache.get(nodeId);
  if (cached) return cached;
  const edges = await api.mctsGetChildren(nodeId);
  childrenCache.set(nodeId, edges);
  for (const e of edges) {
    if (!summaries.has(e.child_id)) summaries.set(e.child_id, edgeSummary(e));
  }
  return edges;
}

function resetView() {
  treeTransform.scale = 1;
  treeTransform.tx = 0;
  treeTransform.ty = 0;
}

async function refresh(): Promise<void> {
  try {
    const info = await api.mctsGetRoot();
    store.rootInfo = info;
    store.expanded = new Set([info.root.id]);
    childrenCache.clear();
    summaries.clear();
    summaries.set(info.root.id, { n: info.root.n, q: info.root.q });
    resetView();
    await ensureChildren(info.root.id);
  } catch (e) {
    store.rootInfo = null;
    toast.info(String(e));
  }
}

async function toggle(nodeId: number) {
  if (store.expanded.has(nodeId)) {
    store.expanded.delete(nodeId);
  } else {
    try {
      await ensureChildren(nodeId);
      store.expanded.add(nodeId);
    } catch (e) {
      console.error('mcts_get_children failed:', e);
    }
  }
}

async function search() {
  store.searching = true;
  try {
    await api.mctsSearch();
    await refresh();
  } catch (e) {
    toast.error('MCTS 搜索失败：' + e);
  } finally {
    store.searching = false;
  }
}

async function fetchDetail(nodeId: number): Promise<MctsNodeDetail | null> {
  try {
    return await api.mctsGetNodeDetail(nodeId);
  } catch {
    return null;
  }
}

function buildLayout(): MctsLayoutNode | null {
  const info = store.rootInfo;
  if (!info) return null;

  function build(id: number, depth: number, edge: MctsEdge | null): MctsLayoutNode {
    const summary = summaries.get(id) ?? { n: 0, q: 0 };
    const node: MctsLayoutNode = { id, depth, edge, n: summary.n, q: summary.q, children: [], hiddenCount: 0, x: 0, y: depth };
    if (store.expanded.has(id)) {
      const edges = childrenCache.get(id) ?? [];
      const shown = store.showAll ? edges : edges.slice(0, MCTS_MAX_CHILDREN);
      node.hiddenCount = edges.length - shown.length;
      node.children = shown.map((e) => build(e.child_id, depth + 1, e));
    }
    return node;
  }

  const leafCounter = { value: 0 };
  function assign(node: MctsLayoutNode): MctsLayoutNode {
    if (node.children.length === 0) {
      node.x = (leafCounter.value += 1) - 0.5;
    } else {
      node.children.forEach(assign);
      node.x = (node.children[0].x + node.children[node.children.length - 1].x) / 2;
    }
    return node;
  }

  return assign(build(info.root.id, 0, null));
}

function collectNodes(root: MctsLayoutNode): MctsLayoutNode[] {
  const out: MctsLayoutNode[] = [];
  (function walk(n: MctsLayoutNode) {
    out.push(n);
    n.children.forEach(walk);
  })(root);
  return out;
}

export function useMctsTree() {
  return {
    store,
    treeTransform,
    refresh,
    toggle,
    search,
    fetchDetail,
    buildLayout,
    collectNodes,
    hasChildren(id: number) {
      return store.expanded.has(id) || (childrenCache.get(id)?.length ?? 0) > 0;
    },
  };
}
