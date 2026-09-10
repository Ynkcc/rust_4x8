import { reactive } from 'vue';

export type ToastType = 'info' | 'success' | 'error';

export interface ToastItem {
  id: number;
  text: string;
  type: ToastType;
}

let nextId = 0;
const toasts = reactive<ToastItem[]>([]);

function push(text: string, type: ToastType, duration: number) {
  const id = ++nextId;
  toasts.push({ id, text, type });
  if (toasts.length > 5) toasts.shift();
  window.setTimeout(() => {
    const idx = toasts.findIndex((t) => t.id === id);
    if (idx >= 0) toasts.splice(idx, 1);
  }, duration);
}

export function useToast() {
  return {
    toasts,
    info: (text: string, duration = 2600) => push(text, 'info', duration),
    success: (text: string, duration = 2600) => push(text, 'success', duration),
    error: (text: string, duration = 4200) => push(text, 'error', duration),
  };
}
