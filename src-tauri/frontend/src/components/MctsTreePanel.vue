<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { MctsNodeDetail } from '../api/types';
import { MCTS_MAX_CHILDREN, useMctsTree } from '../composables/useMctsTree';
import { useGame } from '../composables/useGame';

const emit = defineEmits<{ close: [] }>();

const { store, treeTransform, refresh, toggle, search, fetchDetail, buildLayout, collectNodes, hasChildren } =
  useMctsTree();
const { store: game } = useGame();

// mcts-viz 风格：矩形节点 + 正交折线连线
const NW = 116;
const NH = 64;
const DX = 138;
const DY = 128;
const PAD = 70;

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

function nodeTitle(n: { id: number; edge: { is_chance: boolean; chance_prob: number; action: number } | null }): string {
  if (rootInfo.value && n.id === rootInfo.value.root.id) return 'ROOT';
  if (!n.edge) return '';
  return n.edge.is_chance ? `翻 ${(n.edge.chance_prob * 100).toFixed(0)}%` : `着法 #${n.edge.action}`;
}

function edgePath(n: { x: number; y: number }, c: { x: number; y: number }): string {
  const y1 = py(n) + NH / 2;
  const y2 = py(c) - NH / 2;
  const mid = (y1 + y2) / 2;
  return `M ${px(n)} ${y1} L ${px(n)} ${mid} L ${px(c)} ${mid} L ${px(c)} ${y2}`;
}

function edgeMidY(n: { y: number }, c: { y: number }): number {
  return (py(n) + NH / 2 + (py(c) - NH / 2)) / 2;
}

function isChosenEdge(n: { edge: { is_chance: boolean; action: number } | null }): boolean {
  const info = store.rootInfo;
  return !!info && !!n.edge && !n.edge.is_chance && n.edge.action === info.chosen_action;
}

function edgeWidth(n: { edge: { prior: number; is_chance: boolean } | null }): number {
  if (!n.edge || n.edge.is_chance) return 1.5;
  return Math.max(1.5, Math.min(7, (n.edge?.prior ?? 0) * 40));
}

function edgeStroke(n: { edge: { is_chance: boolean } | null }): string {
  return n.edge?.is_chance ? '#c9a24b' : '#9a8f7f';
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
          <!-- 正交折线连线（mcts-viz 风格） -->
          <template v-for="n in nodes" :key="`edges-${n.id}`">
            <template v-for="c in n.children" :key="`e-${n.id}-${c.id}`">
              <path
                :d="edgePath(n, c)"
                fill="none"
                :stroke="isChosenEdge(c) ? '#d32f2f' : edgeStroke(c)"
                :stroke-width="isChosenEdge(c) ? 4.5 : edgeWidth(c)"
                :stroke-dasharray="c.edge?.is_chance ? '5 4' : undefined"
              />
              <text
                v-if="(c.edge?.prior ?? 0) > 0"
                :x="px(c) + 5"
                :y="edgeMidY(n, c) - 4"
                font-size="10"
                fill="#8a8378"
              >
                P={{ c.edge!.prior.toFixed(2) }} N={{ c.n }}
              </text>
            </template>
          </template>

          <!-- 矩形节点 -->
          <g
            v-for="n in nodes"
            :key="`n-${n.id}`"
            :transform="`translate(${px(n)}, ${py(n)})`"
            class="mcts-node"
            @click.stop="toggle(n.id)"
            @mouseenter="onNodeEnter(n.id, $event)"
            @mouseleave="onNodeLeave"
          >
            <rect
              class="mcts-node-rect"
              :x="-NW / 2"
              :y="-NH / 2"
              :width="NW"
              :height="NH"
              rx="8"
              :class="{ 'is-chosen': isChosenEdge(n) }"
            />
            <text :y="-6" text-anchor="middle" class="mcts-node-title">
              {{ nodeTitle(n) }}
            </text>
            <text :y="14" text-anchor="middle" class="mcts-node-stats">
              N={{ n.n }} Q={{ n.q.toFixed(2) }}
            </text>
            <g v-if="hasChildren(n.id)" class="mcts-collapse">
              <circle :cy="NH / 2" r="9" />
              <text :y="NH / 2 + 4" text-anchor="middle">
                {{ store.expanded.has(n.id) ? '−' : '+' }}
              </text>
            </g>
          </g>

          <template v-for="n in nodes" :key="`hidden-${n.id}`">
            <text
              v-if="store.expanded.has(n.id) && n.hiddenCount > 0"
              :x="px(n)"
              :y="py(n) + DY - 24"
              text-anchor="middle"
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
