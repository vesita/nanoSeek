"""vDB: 纯内容寻址的向量存储引擎 (Python 参考实现, 预研尝试期)。

定位 (v3, 纯向量版):
  数据无关 / 不参与训练 / 确定性 / 只经 f64 数组。
  模型只产出向量: 写向量写 / 查向量查 (topk)。
  不再有 key-value 二重表 —— 条目即向量, 查询即向量, 寻址即相似度。
  指令与路由 (旁路/分区/topk 预算) 由桥接层的有限 n-bit 指令决定 (见 interface/)。

分区 (用户已定):
  fast 高速区 O(1)      : 向量 -> 量化 -> 哈希 (开放寻址), 热数据/工作记忆。
  slow 低速区 O(n)/O(n log n) : 连续 np 数组批量 matmul 精确 topk (尝试期),
                          或排序索引; Rust 版用 sorted/BTree/SIMD 达标称复杂度。

遗忘 (引擎周期功能, 不占指令位):
  score = exp(-dt/tau) * (1 + alpha*access)
  fast 满 -> 最低分「降级」进 slow (热->冷); slow 满 -> 淘汰; ttl 过期 -> 回收。

确定性: 无随机源; 同指令序列 => 同状态 (RL 可重放)。
"""
import numpy as np

_FNV_OFF = 14695981039346656037
_FNV_P = 1099511628211
_M64 = (1 << 64) - 1


def _quant_bytes(key, scale=127):
    k = np.asarray(key, dtype=np.float64).reshape(-1)
    m = max(float(np.abs(k).max()), 1e-9)
    return np.rint(np.clip(k / m, -1.0, 1.0) * scale).astype(np.int8).tobytes()


def _hash(b, cap):
    h = _FNV_OFF
    for x in b:
        h = ((h ^ x) * _FNV_P) & _M64
    return h % cap


class VDB:
    def __init__(self, d_key=32, d_val=64,
                 fast_cap=1024, slow_cap=131072,
                 tau=512.0, ttl=4096.0, alpha=0.05, n_sig=16):
        self.d_key, self.d_val = int(d_key), int(d_val)
        self.fast_cap, self.slow_cap = int(fast_cap), int(slow_cap)
        self.tau, self.ttl, self.alpha = float(tau), float(ttl), float(alpha)
        self.n_sig = min(int(n_sig), int(d_key))
        self._reset()

    def _reset(self):
        c, s, k, v = self.fast_cap, self.slow_cap, self.d_key, self.d_val
        self._fk = np.zeros((c, k)); self._fv = np.zeros((c, v))
        self._ft = np.full(c, -1e18); self._fa = np.zeros(c)
        self._focc = np.zeros(c, bool); self._fmap = {}
        self._sk = np.zeros((s, k)); self._sv = np.zeros((s, v))
        self._st = np.full(s, -1e18); self._sa = np.zeros(s)
        self._socc = np.zeros(s, bool)

    # ---------------- 指令: write ----------------
    def write(self, key, value, t, region=0, overwrite=False):
        '''region: 0=fast 1=slow。overwrite=False 且键已存在时不覆盖(flag位控制)。'''
        kb = _quant_bytes(key)
        k = np.asarray(key, np.float64).reshape(-1)
        vv = np.asarray(value, np.float64).reshape(-1)
        if region == 0:
            # KV 通道: 签名位即地址 (精确)
            return self._w_fast(_quant_bytes(k[:self.n_sig]), k, vv, float(t), overwrite)
        # Vec 通道: 内容即地址 (相似度)
        return self._w_slow(kb, k, vv, float(t), overwrite)

    def _w_fast(self, kb, k, vv, t, overwrite):
        s = self._fmap.get(kb)
        if s is not None:
            if not overwrite and self._focc[s]:
                self._ft[s] = t; self._fa[s] += 1.0
                return s
            self._fv[s] = vv; self._ft[s] = t; self._fa[s] += 1.0
            return s
        free = np.flatnonzero(~self._focc)
        if free.size == 0:
            self._demote_victim(t)
            free = np.flatnonzero(~self._focc)
        s = int(free[0])
        self._fk[s] = k; self._fv[s] = vv
        self._ft[s] = t; self._fa[s] = 1.0; self._focc[s] = True
        self._fmap[kb] = s
        return s

    def _w_slow(self, kb, k, vv, t, overwrite):
        ev = -1
        hit = np.flatnonzero(self._socc & (np.abs(self._sk - k).sum(axis=1) < 1e-9))
        if hit.size:
            s = int(hit[0])
            if overwrite:
                self._sv[s] = vv
            self._st[s] = t; self._sa[s] += 1.0
            return s
        free = np.flatnonzero(~self._socc)
        if free.size == 0:
            sc = self._score_slow(np.full(self.slow_cap, t))
            ev = int(np.argmin(np.where(self._socc, sc, np.inf)))
            self._socc[ev] = False; free = [ev]
        s = int(np.flatnonzero(~self._socc)[0]) if np.any(~self._socc) else int(ev)
        self._sk[s] = k; self._sv[s] = vv
        self._st[s] = t; self._sa[s] = 1.0; self._socc[s] = True
        return s

    # ---------------- 指令: read (topk, K 由指令位给) ----------------
    def read(self, query, t, topk=1, region=0):
        t = float(t)
        q = np.asarray(query, np.float64).reshape(-1)
        if region == 0:
            # KV 通道: 签名字节直接命中槽
            s = self._fmap.get(_quant_bytes(q[:self.n_sig]))
            if s is None or not self._focc[s]:
                return []
            self._ft[s] = t; self._fa[s] += 1.0
            return [self._fv[s].copy()]
        if not self._socc.any():
            return []
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = np.where(self._socc, self._sk @ qn, -np.inf)
        idx = np.argpartition(-sims, min(topk, self.slow_cap - 1))[:topk]
        idx = idx[np.isfinite(sims[idx])]
        for i in idx:
            self._st[i] = t; self._sa[i] += 1.0
        return [self._sv[i].copy() for i in idx]

    # ---------------- 遗忘 (周期功能) ----------------
    def _score_fast(self, t):
        return np.exp(-np.clip(t - self._ft, 0, None) / self.tau) * (1.0 + self.alpha * self._fa)

    def _score_slow(self, t):
        return np.exp(-np.clip(t - self._st, 0, None) / self.tau) * (1.0 + self.alpha * self._sa)

    def _demote_victim(self, t):
        '''fast 满: 最低分降级进 slow (热->冷)。'''
        sc = np.where(self._focc, self._score_fast(t), np.inf)
        i = int(np.argmin(sc))
        self._w_slow(_quant_bytes(self._fk[i]), self._fk[i].copy(),
                     self._fv[i].copy(), self._ft[i], overwrite=True)
        self._focc[i] = False; self._ft[i] = -1e18; self._fa[i] = 0.0
        for kb, s in list(self._fmap.items()):
            if s == i:
                del self._fmap[kb]

    def forget(self, t):
        '''周期回收: ttl 过期直接清除。返回清除条数 (fast+slow)。'''
        t = float(t)
        n = 0
        fexp = self._focc & ((t - self._ft) > self.ttl)
        for i in np.flatnonzero(fexp):
            self._focc[i] = False; self._ft[i] = -1e18; self._fa[i] = 0.0
            n += 1
        for kb, s in list(self._fmap.items()):
            if not self._focc[s]:
                del self._fmap[kb]
        sexp = self._socc & ((t - self._st) > self.ttl)
        n += int(sexp.sum())
        self._socc &= ~sexp; self._st = np.where(sexp, -1e18, self._st)
        self._sa = np.where(sexp, 0.0, self._sa)
        return n

    # ---------------- 状态快照 (RL 重放用) ----------------
    def get_state(self):
        return {'fk': self._fk.copy(), 'fv': self._fv.copy(), 'ft': self._ft.copy(),
                'fa': self._fa.copy(), 'focc': self._focc.copy(),
                'sk': self._sk.copy(), 'sv': self._sv.copy(), 'st': self._st.copy(),
                'sa': self._sa.copy(), 'socc': self._socc.copy()}

    def set_state(self, st):
        self._fk = st['fk'].copy(); self._fv = st['fv'].copy()
        self._ft = st['ft'].copy(); self._fa = st['fa'].copy()
        self._focc = st['focc'].copy(); self._fmap = {}
        self._sk = st['sk'].copy(); self._sv = st['sv'].copy()
        self._st = st['st'].copy(); self._sa = st['sa'].copy()
        self._socc = st['socc'].copy()
        for i in np.flatnonzero(self._focc):
            self._fmap[_quant_bytes(self._fk[i])] = int(i)

    def reset(self):
        self._reset()

    # ---------------- KV+Vec 混合检索: 签名过滤 -> 相似度 topk ----------------
    def read_hybrid(self, query, t, topk=1, n_sig=None):
        n = self.n_sig if n_sig is None else int(n_sig)
        q = np.asarray(query, np.float64).reshape(-1)
        if not self._socc.any():
            return []
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = np.where(self._socc, self._sk @ qn, -np.inf)
        if n > 0:
            ok = np.abs(self._sk[:, :n] - q[:n]).sum(axis=1) < 1e-6
            sims = np.where(ok, sims, -np.inf)
        idx = np.argpartition(-sims, min(topk, self.slow_cap - 1))[:topk]
        idx = idx[np.isfinite(sims[idx])]
        for i in idx:
            self._st[i] = float(t); self._sa[i] += 1.0
        return [self._sv[i].copy() for i in idx]

    def __len__(self):
        return int(self._focc.sum()) + int(self._socc.sum())


if __name__ == '__main__':
    # 轻量自测 (CPU, 毫秒级; 不是训练, 不占 GPU)
    db = VDB(fast_cap=8, slow_cap=64, tau=8.0, ttl=32.0)
    rng = np.random.default_rng(0)
    hit = 0
    for t in range(200):
        k = rng.standard_normal(db.d_key)
        v = np.tanh(rng.standard_normal(db.d_val))
        db.write(k, v, t=float(t))
        if t % 5 == 0:
            r = db.read(k, t=float(t), topk=1, region=0)
            r2 = db.read_hybrid(k, t=float(t), topk=1)
            r3 = db.read(k, t=float(t), topk=3, region=1)
            if r:
                hit += 1
        if t % 10 == 0:
            db.forget(t=float(t))
    assert hit > 0 and len(db) > 0 and len(db) <= db.fast_cap + db.slow_cap
    print('vdb selftest ok:', 'fast:', int(db._focc.sum()), 'slow:', int(db._socc.sum()), 'hits:', hit)
