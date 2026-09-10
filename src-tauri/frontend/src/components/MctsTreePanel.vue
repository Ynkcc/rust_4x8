<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { MctsNodeDetail } from '../api/types';
import { MCTS_MAX_CHILDREN, useMctsTree } from '../composables/useMctsTree';
import { useGame } from '../composables/useGame';

const emit = defineEmits<{ close: [] }>();

const { store, treeTransform, refresh, toggle, search, fetchDetail, buildLayout, collectNodes, hasChildren } =
  useMctsTree();
const { store: game } = useGame();

const DX = 92;
const DY = 96;
const PAD = 60;

const wrapEl = ref<HTMLElement | null>(null);
const tooltip = ref<{ x: number; y: number; detail: MctsNodeDetail } | null>(null);

const layout = computed(() => buildLayout());
const nodes = computed(() => (layout.value ? collectNodes(layout.value) : []));

const svgViewBox = computed(() => {
  const leafCount = Math.max(1, nodes.value.filter((n) => n.children.length === 0).length);
  const maxDepth = Math.max(0, ...nodes.value.map((n) => n.depth));
  return { width: leafCount * DX + PAD * 2, height: (maxDepth + 1) * DY + PAD * 2 };
});

const rootInfo = computed(() => store.rootInfo);

const px = (n: { x: number }) => PAD + n.x * DX;
const py = (n: { y: number }) => PAD + n.y * DY;

function qColor(q: number): string {
  const t = Math.max(-1, Math.min(1, q));
  const hue = t >= 0 ? 215 : 5;
  const sat = 15 + Math.abs(t) * 65;
  const light = 82 - Math.abs(t) * 22;
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}

function edgeLabel(edge: { is_chance: boolean; chance_prob: number; action: number } | null): string {
  if (!edge) return '';
  return edge.is_chance ? `翻${(edge.chance_prob * 100).toFixed(0)}%` : `#${edge.action}`;
}

function isChosenEdge(n: { edge: { is_chance: boolean; action: number } | null }): boolean {
  const info = store.rootInfo;
  return !!info && !!n.edge && !n.edge.is_chance && n.edge.action === info.chosen_action;
}

function nodeRadius(n: { n: number }): number {
  return 5 + Math.sqrt(n.n || 0) * 1.6;
}

function edgeWidth(prior: number): number {
  return Math.max(1, Math.min(8, prior * 40));
}

function edgeStroke(n: { edge: { prior: number; is_chance: boolean } | null }): string {
  return n.edge?.is_chance ? 'transparent' : '#9a8f7f';
}

async function onNodeEnter(nodeId: number, evt: MouseEvent) {
  const detail = await fetchDetail(nodeId);
  if (!detail || !wrapEl.value) return;
  const rect = wrapEl.value.getBoundingClientRect();
  let x = evt.clientX - rect.left + 14;
  let y = evt.clientY - rect.top + 14;
  if (x + 190 > rect.width) x -= 210;
  if (y + 120 > rect.height) y -= 130;
  tooltip.value = { x, y, detail };
}

function onNodeLeave() {
  tooltip.value = null;
}

function onWheel(evt: WheelEvent) {
  evt.preventDefault();
  const factor = evt.deltaY < 0 ? 1.15 : 1 / 1.15;
  const rect = wrapEl.value?.getBoundingClientRect();
  if (!rect) return;
  const cx = evt.clientX - rect.left;
  const cy = evt.clientY - rect.top;
  treeTransform.tx = cx - (cx - treeTransform.tx) * factor;
  treeTransform.ty = cy - (cy - treeTransform.ty) * factor;
  treeTransform.scale = Math.max(0.2, Math.min(4, treeTransform.scale * factor));
}

let dragging = false;
let lastX = 0;
let lastY = 0;

function onDragStart(evt: MouseEvent) {
  if ((evt.target as HTMLElement).closest('g')) return;
  dragging = true;
  lastX = evt.clientX;
  lastY = evt.clientY;
}
function onDragMove(evt: MouseEvent) {
  if (!dragging) return;
  treeTransform.tx += evt.clientX - lastX;
  treeTransform.ty += evt.clientY - lastY;
  lastX = evt.clientX;
  lastY = evt.clientY;
}
function onDragEnd() {
  dragging = false;
}

// 打开面板（visible 变 true）或棋盘更新且面板开启时刷新树
watch(
  () => game.state,
  () => {
    if (store.rootInfo) void refresh();
  },
);

function transformAttr(): string {
  return `translate(${treeTransform.tx}, ${treeTransform.ty}) scale(${treeTransform.scale})`;
}
</script>

<template>
  <aside class="sidebar mcts-sidebar">
    <header class="sidebar-header">
      <h3>MCTS 搜索树</h3>
      <button class="icon-button" aria-label="关闭" @click="emit('close')">×</button>
    </header>
    <div class="mcts-toolbar">
      <button :disabled="store.searching" @click="search">
        {{ store.searching ? '搜索中…' : '重新搜索' }}
      </button>
      <button @click="refresh">刷新</button>
      <label><input v-model="store.showAll" type="checkbox" /> 显示全部子节点</label>
      <span v-if="rootInfo" class="mcts-info">
        根 N={{ rootInfo.root.n }} Q={{ rootInfo.root.q.toFixed(3) }} 选择动作 #{{
          rootInfo.chosen_action
        }}
      </span>
    </div>
    <div
      ref="wrapEl"
      class="mcts-svg-wrap"
      @wheel="onWheel"
      @mousedown="onDragStart"
      @mousemove="onDragMove"
      @mouseup="onDragEnd"
      @mouseleave="onDragEnd"
    >
      <div v-if="!rootInfo" class="mcts-empty">
        暂无搜索树：请先让 MctsDL / MctsOnnx 对手走一步，或手动搜索
      </div>
      <svg v-else :viewBox="`0 0 ${svgViewBox.width} ${svgViewBox.height}`" width="100%" height="100%">
        <g :transform="transformAttr()">
          <template v-for="n in nodes" :key="`edges-${n.id}`">
            <template v-for="c in n.children" :key="`e-${n.id}-${c.id}`">
              <path
                :d="`M ${px(n)} ${py(n)} C ${px(n)} ${(py(n) + py(c)) / 2}, ${px(c)} ${(py(n) + py(c)) / 2}, ${px(c)} ${py(c)}`"
                fill="none"
                :stroke="isChosenEdge(c) ? '#2f9e44' : edgeStroke(c)"
                :stroke-width="isChosenEdge(c) ? 5 : edgeWidth(c.edge?.prior ?? 0)"
                :stroke-dasharray="c.edge?.is_chance ? '5 4' : undefined"
              />
              <text
                v-if="(c.edge?.prior ?? 0) > 0"
                :x="(px(n) + px(c)) / 2 + 6"
                :y="(py(n) + py(c)) / 2"
                font-size="10"
                fill="#8a8378"
              >
                P={{ c.edge!.prior.toFixed(2) }} N={{ c.n }}
              </text>
            </template>
          </template>

          <g
            v-for="n in nodes"
            :key="`n-${n.id}`"
            :transform="`translate(${px(n)}, ${py(n)})`"
            class="mcts-node"
            @click.stop="toggle(n.id)"
            @mouseenter="onNodeEnter(n.id, $event)"
            @mouseleave="onNodeLeave"
          >
            <circle
              :r="nodeRadius(n)"
              :fill="qColor(n.q)"
              :stroke="n.id === rootInfo!.root.id ? '#4a3f2f' : '#6b6152'"
              :stroke-width="n.id === rootInfo!.root.id ? 3 : 1.5"
            />
            <text :y="-nodeRadius(n) - 5" text-anchor="middle" font-size="11" fill="#4a3f2f">
              {{ n.id === rootInfo!.root.id ? 'ROOT' : edgeLabel(n.edge) }}
            </text>
            <text :y="nodeRadius(n) + 13" text-anchor="middle" font-size="10" fill="#6b6152">
              N={{ n.n }} Q={{ n.q.toFixed(2) }}
            </text>
            <text
              v-if="hasChildren(n.id)"
              :x="nodeRadius(n) + 4"
              y="4"
              font-size="11"
              fill="#2f6f9e"
            >
              {{ store.expanded.has(n.id) ? '−' : '+' }}
            </text>
          </g>

          <template v-for="n in nodes" :key="`hidden-${n.id}`">
            <text
              v-if="store.expanded.has(n.id) && n.hiddenCount > 0"
              :x="px(n) - 40"
              :y="py(n) + DY - 18"
              font-size="10"
              fill="#b0651f"
            >
              …还有 {{ n.hiddenCount }} 个子节点（前 {{ MCTS_MAX_CHILDREN }} 个）
            </text>
          </template>
        </g>
      </svg>
      <div
        v-if="tooltip"
        class="mcts-tooltip"
        :style="{ left: tooltip.x + 'px', top: tooltip.y + 'px' }"
      >
        <template v-if="tooltip.detail">
          <div>
            节点 #{{ tooltip.detail.id }}（{{ tooltip.detail.player === 'Red' ? '红' : '黑' }}方）
          </div>
          <div v-if="tooltip.detail.is_chance">【机会节点】</div>
          <div v-if="tooltip.detail.is_terminal">【终局】</div>
          <div>
            N={{ tooltip.detail.n }} Q={{ tooltip.detail.q.toFixed(3) }} Q_hp={{
              tooltip.detail.health_q.toFixed(3)
            }}
          </div>
          <div>
            prior={{ tooltip.detail.prior.toFixed(4) }} logit={{ tooltip.detail.logit.toFixed(3) }}
          </div>
          <div>V先验={{ tooltip.detail.initial_value.toFixed(3) }}</div>
          <div>
            子节点={{ tooltip.detail.child_count }} 机会结果={{ tooltip.detail.outcome_count }}
          </div>
        </template>
      </div>
    </div>
  </aside>
</template>
