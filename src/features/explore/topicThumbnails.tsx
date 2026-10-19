import type { ReactNode } from "react";

// Shared vocabulary: accent = the key element, muted = context, surface = node fills. All 140x80.
const a = "var(--accent)";
const mu = "var(--muted)";
const S = "var(--surface)";
const mf = "color-mix(in srgb, var(--muted) 45%, transparent)";
const m = { stroke: mu, strokeOpacity: 0.6, fill: "none" };
const ac = { stroke: a, fill: "none", strokeWidth: 1.5 };
const T = { fill: mu, fontSize: 9, fontFamily: "ui-monospace, monospace", textAnchor: "middle" as const };
type P = number[][];

const dots = (p: P, fill = a, r = 2.5) => p.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={r} fill={fill} />);
const boxes = (p: P, stroke = mu, fill = S) => p.map(([x, y, w, h], i) => <rect key={i} x={x} y={y} width={w} height={h} rx={2} fill={fill} stroke={stroke} />);
const bars = (x0: number, step: number, w: number, hs: number[], fill: (i: number) => string, base = 70) =>
  hs.map((h, i) => <rect key={i} x={x0 + i * step} y={base - h} width={w} height={h} rx={1} fill={fill(i)} />);
/** Sampled path of y = f(x). */
const curve = (f: (x: number) => number, x0 = 10, x1 = 130) => {
  let d = "";
  for (let x = x0; x <= x1; x += 2) d += `${d ? "L" : "M"}${x} ${f(x).toFixed(1)}`;
  return d;
};
const gauss = (c: number, sd: number, h: number, base = 70) => (x: number) => base - h * Math.exp(-((x - c) ** 2) / (2 * sd * sd));
const check = (x: number, y: number, stroke = a) => <path d={`M${x - 4} ${y} l3 3 l5 -6`} stroke={stroke} strokeWidth={1.8} fill="none" />;
const cross = (x: number, y: number, stroke = mu) => <path d={`M${x - 3} ${y - 3} l6 6 M${x + 3} ${y - 3} l-6 6`} stroke={stroke} strokeWidth={1.8} />;

export const THUMBNAILS: Record<string, () => ReactNode> = {
  "forward-propagation": () => {
    const L = [[20, 25], [20, 55]], M = [[70, 15], [70, 40], [70, 65]], R = [[120, 40]];
    return (<>
      {L.flatMap(([x1, y1]) => M.map(([x2, y2]) => <line key={`${x1}${y1}${x2}${y2}`} x1={x1} y1={y1} x2={x2} y2={y2} {...m} />))}
      {M.map(([x1, y1]) => <line key={y1} x1={x1} y1={y1} x2={120} y2={40} stroke={a} strokeWidth={1.5} />)}
      {[...L, ...M, ...R].map(([x, y]) => <circle key={`${x}-${y}`} cx={x} cy={y} r={6} fill={S} stroke={a} />)}
    </>);
  },
  backpropagation: () => {
    const N = [[25, 20], [25, 52], [70, 14], [70, 36], [70, 58], [115, 36]];
    return (<>
      {[0, 1].flatMap((i) => [2, 3, 4].map((j) => <line key={`${i}${j}`} x1={N[i][0]} y1={N[i][1]} x2={N[j][0]} y2={N[j][1]} {...m} />))}
      {[2, 3, 4].map((j) => <line key={j} x1={N[j][0]} y1={N[j][1]} x2={115} y2={36} stroke={a} strokeDasharray="3 2" />)}
      {N.map(([x, y]) => <circle key={`${x}${y}`} cx={x} cy={y} r={5} fill={S} stroke={a} />)}
      <path d="M118 72 H24 M30 68 l-6 4 l6 4" {...ac} />
    </>);
  },
  "neurons-and-activation-functions": () => (<>
    {[20, 40, 60].map((y) => <line key={y} x1={12} y1={y} x2={58} y2={40} {...m} />)}
    {dots([[12, 20], [12, 40], [12, 60]], mu)}
    <circle cx={60} cy={40} r={11} fill={S} stroke={a} strokeWidth={1.5} /><line x1={71} y1={40} x2={86} y2={40} {...m} />
    <path d="M90 66 H132 M92 14 V68" {...m} /><path d="M92 56 H112 L130 18" {...ac} />
  </>),
  overfitting: () => (<>
    {dots([[15, 55], [30, 40], [45, 30], [60, 35], [75, 45], [90, 52], [105, 40], [120, 25]])}
    <path d="M10 60 C 40 15, 70 60, 90 50 S 115 20, 130 22" {...ac} />
    <path d="M10 58 L 130 28" {...m} strokeDasharray="3 3" />
  </>),
  "linear-regression": () => {
    const yl = (x: number) => 66 - ((x - 10) * 5) / 12;
    const pts = [[20, 54], [38, 60], [55, 42], [75, 48], [95, 28], [112, 30], [125, 12]];
    return (<>
      {pts.map(([x, y]) => { const d = Math.abs(y - yl(x)); return <rect key={x} x={x} y={Math.min(y, yl(x))} width={d} height={d} fill={a} fillOpacity={0.15} stroke={a} strokeOpacity={0.5} />; })}
      <line x1={10} y1={66} x2={130} y2={16} stroke={a} strokeWidth={1.5} />
      {dots(pts, mu)}
    </>);
  },
  "logistic-regression": () => (<>
    <path d={curve((x) => 66 - 52 / (1 + Math.exp(-(x - 70) / 9)))} {...ac} />
    <line x1={70} y1={8} x2={70} y2={72} {...m} strokeDasharray="3 3" />
    {dots([[16, 66], [28, 66], [40, 66], [52, 66], [82, 66]], mu)}{dots([[58, 14], [88, 14], [100, 14], [112, 14], [124, 14]])}
  </>),
  "loss-functions": () => (<>
    <path d="M10 72 H130" {...m} />
    <path d={curve((x) => 70 - (x - 70) ** 2 / 62)} {...ac} />
    <path d={curve((x) => 70 - Math.abs(x - 70) * 0.95)} {...m} strokeDasharray="3 2" />
    <circle cx={128} cy={14} r={3} fill={a} />
  </>),
  "gradient-descent": () => (<>
    {[[50, 30], [36, 21], [22, 13], [9, 5]].map(([rx, ry]) => <ellipse key={rx} cx={86} cy={42} rx={rx} ry={ry} {...m} />)}
    <path d="M30 20 L50 58 L62 28 L76 52 L80 36 L86 42" {...ac} />
    {dots([[50, 58], [62, 28], [76, 52], [80, 36]], a, 2)}<circle cx={30} cy={20} r={4.5} fill={a} />
  </>),
  "bias-and-variance": () => (<>
    {[38, 102].flatMap((cx) => [28, 18, 8].map((r) => <circle key={`${cx}${r}`} cx={cx} cy={40} r={r} {...m} />))}
    {dots([[50, 26], [54, 29], [52, 23], [56, 26], [49, 30]])}
    {dots([[90, 30], [112, 50], [98, 58], [115, 28], [92, 48], [106, 34]])}
  </>),
  "decision-trees": () => (<>
    <path d="M70 14 L40 38 M70 14 L100 38 M40 38 L25 64 M40 38 L55 64 M100 38 L88 64 M100 38 L114 64" {...m} />
    <path d="M70 14 L100 38 L114 64" {...ac} />
    {boxes([[60, 9, 20, 10], [30, 33, 20, 10], [90, 33, 20, 10]], a)}
    {[25, 55, 88, 114].map((x, i) => <rect key={x} x={x - 6} y={60} width={12} height={10} rx={2} fill={i % 2 ? a : mf} />)}
  </>),
  "supervised-vs-unsupervised-learning": () => (<>
    <line x1={70} y1={6} x2={70} y2={74} {...m} />
    {dots([[14, 20], [24, 28], [18, 36], [30, 18]])}{dots([[44, 52], [54, 60], [48, 66], [58, 48]], mu)}
    <line x1={10} y1={62} x2={62} y2={14} {...m} strokeDasharray="3 2" />
    {dots([[84, 20], [94, 28], [88, 36], [100, 18], [114, 52], [124, 60], [118, 66], [128, 48]], mu)}
    <ellipse cx={92} cy={26} rx={14} ry={13} stroke={a} fill="none" strokeDasharray="3 2" /><ellipse cx={121} cy={57} rx={13} ry={13} stroke={a} fill="none" strokeDasharray="3 2" />
  </>),
  "feature-engineering": () => (<>
    {[0, 1, 2, 3].flatMap((c) => [0, 1, 2, 3].map((r) => <rect key={`${c}${r}`} x={10 + c * 14} y={14 + r * 12} width={13} height={11} fill={c === 2 ? a : S} fillOpacity={c === 2 ? 0.35 : 1} stroke={c === 2 ? a : mu} strokeOpacity={0.6} />))}
    <path d="M72 38 H86 M82 34 l4 4 l-4 4" {...m} />
    {bars(94, 14, 10, [16, 28, 42], () => a, 62)}
  </>),
  "train-validation-and-test-splits": () => (<>
    {dots([[20, 12], [34, 14], [50, 11], [66, 13], [82, 12], [112, 13], [126, 12]], mu, 2)}
    <rect x={10} y={26} width={70} height={16} rx={2} fill={mf} />
    <rect x={82} y={26} width={22} height={16} rx={2} fill={a} />
    <rect x={106} y={26} width={24} height={16} rx={2} fill={S} stroke={mu} strokeDasharray="2 2" />
    <text x={45} y={56} {...T} fontSize={8}>train</text><text x={93} y={56} {...T} fill={a} fontSize={8}>val</text><text x={118} y={56} {...T} fontSize={8}>test</text>
  </>),
  "data-leakage": () => (<>
    {boxes([[10, 28, 74, 40], [96, 28, 34, 40]])}
    <text x={47} y={52} {...T}>train</text><text x={113} y={52} {...T}>test</text>
    <path d="M113 26 Q 96 4 64 24" {...ac} strokeDasharray="3 2" /><path d="M68 18 l-4 6 l7 1" {...ac} />
  </>),
  "regression-metrics": () => {
    const h = [10, -8, 14, -6, 4, -12, 9, -4, 30, -7];
    return (<>
      <line x1={8} y1={44} x2={132} y2={44} {...m} />
      {h.map((v, i) => <rect key={i} x={14 + i * 12} y={v > 0 ? 44 - v : 44} width={7} height={Math.abs(v)} rx={1} fill={v === 30 ? a : mf} />)}
    </>);
  },
  "classification-metrics": () => (<>
    <rect x={40} y={8} width={64} height={64} {...m} /><line x1={40} y1={72} x2={104} y2={8} {...m} strokeDasharray="3 3" />
    <path d="M40 72 C 44 30, 60 14, 104 8" {...ac} /><circle cx={54} cy={26} r={4} fill={a} />
  </>),
  "confusion-matrices": () => (<>
    {[[0, 0, "42"], [1, 0, "5"], [0, 1, "3"], [1, 1, "38"]].map(([c, r, n]) => {
      const diag = c === r;
      return (<g key={`${c}${r}`}>
        <rect x={40 + Number(c) * 31} y={8 + Number(r) * 33} width={29} height={31} rx={2} fill={diag ? a : S} fillOpacity={diag ? 0.3 : 1} stroke={diag ? a : mu} strokeOpacity={0.7} />
        <text x={54.5 + Number(c) * 31} y={27 + Number(r) * 33} {...T} fill={diag ? a : mu}>{n}</text>
      </g>);
    })}
  </>),
  "next-token-prediction": () => (<>
    {boxes([[6, 32, 26, 16], [36, 32, 26, 16]])}<text x={19} y={43} {...T}>the</text><text x={49} y={43} {...T}>cat</text>
    <path d="M66 40 H72" {...m} />
    {["sat", "ran", "is", "on"].map((w, i) => <text key={w} x={86} y={18 + i * 16} {...T}>{w}</text>)}
    {[40, 24, 14, 8].map((w, i) => <rect key={w} x={98} y={11 + i * 16} width={w} height={9} rx={1} fill={i ? mf : a} />)}
  </>),
  tokenization: () => (<>
    <text x={70} y={16} {...T} fontSize={10}>unbelievable</text>
    {[[14, 24, "un", "403"], [42, 46, "believ", "11912"], [92, 34, "able", "481"]].map(([x, w, t, id]) => (<g key={t}>
      <rect x={x} y={28} width={w} height={18} rx={3} fill={a} fillOpacity={0.15} stroke={a} />
      <text x={Number(x) + Number(w) / 2} y={41} {...T} fill={a} fontSize={10}>{t}</text>
      <text x={Number(x) + Number(w) / 2} y={64} {...T} fontSize={8}>{id}</text>
    </g>))}
  </>),
  "temperature-and-sampling": () => (<>
    {bars(10, 9, 7, [52, 16, 9, 6, 3], () => mf)}
    <path d="M62 40 H76 M72 36 l4 4 l-4 4" {...m} />
    {bars(86, 9, 7, [26, 22, 18, 15, 12], () => a)}
  </>),
  embeddings: () => (<>
    {dots([[30, 20], [40, 30], [25, 35], [100, 55], [110, 45], [95, 60], [80, 20]], mu, 3)}
    <line x1={70} y1={70} x2={40} y2={30} stroke={a} strokeDasharray="3 2" /><line x1={70} y1={70} x2={30} y2={20} stroke={a} strokeDasharray="3 2" />
    <path d="M66 66 l8 8 M74 66 l-8 8" stroke={a} strokeWidth={2} />
  </>),
  "positional-information": () => (<>
    <path d={curve((x) => 16 + 7 * Math.sin(x / 3))} {...m} />
    <path d={curve((x) => 40 + 8 * Math.sin(x / 7))} {...ac} />
    <path d={curve((x) => 64 + 8 * Math.sin(x / 15))} {...m} />
  </>),
  "self-attention": () => {
    const xs = [20, 45, 70, 95, 120];
    return (<>
      {xs.map((x, i) => <line key={x} x1={70} y1={62} x2={x} y2={18} stroke={a} strokeWidth={[1, 3.5, 0.6, 2, 1][i]} strokeOpacity={0.8} />)}
      {xs.flatMap((x) => [14, 66].map((y) => <circle key={`${x}${y}`} cx={x} cy={y} r={5} fill={S} stroke={y === 66 && x === 70 ? a : mu} />))}
    </>);
  },
  "query-key-and-value": () => (<>
    {[16, 28, 40, 52].map((y) => <rect key={y} x={10} y={y} width={12} height={11} fill={a} fillOpacity={0.3} stroke={a} />)}
    {[19, 41, 63].map((y) => <line key={y} x1={24} y1={40} x2={68} y2={y} {...m} />)}
    {["Q", "K", "V"].map((l, r) => (<g key={l}>
      <text x={76} y={22 + r * 22} {...T} fill={a}>{l}</text>
      {[0, 1, 2, 3].map((c) => <rect key={c} x={86 + c * 11} y={14 + r * 22} width={10} height={10} fill={S} stroke={mu} />)}
    </g>))}
  </>),
  "transformer-blocks": () => (<>
    <line x1={4} y1={40} x2={136} y2={40} {...m} />
    <path d="M6 40 V18 H58 V35 M63 40 V64 H118 V45" {...m} strokeDasharray="3 2" />
    {boxes([[12, 32, 10, 16], [70, 32, 10, 16]])}{boxes([[27, 30, 24, 20], [86, 30, 24, 20]], a)}
    {[58, 118].map((x) => <g key={x}><circle cx={x} cy={40} r={5} fill={S} stroke={a} /><path d={`M${x - 3} 40 h6 M${x} 37 v6`} stroke={a} /></g>)}
  </>),
  "fine-tuning-fundamentals": () => (<>
    {[0, 1, 2, 3].flatMap((c) => [0, 1, 2, 3].map((r) => <rect key={`${c}${r}`} x={8 + c * 11} y={18 + r * 11} width={9} height={9} fill={mf} />))}
    <text x={60} y={44} {...T} fontSize={12}>+</text>
    <rect x={68} y={18} width={6} height={42} rx={1} fill={a} /><rect x={80} y={18} width={52} height={6} rx={1} fill={a} />
    <path d={curve((x) => 66 - 32 * Math.exp(-(x - 82) / 10), 82, 132)} {...ac} />
  </>),
  "chunking-strategies": () => (<>
    <rect x={20} y={6} width={96} height={68} rx={2} fill={S} stroke={mu} strokeOpacity={0.6} />
    {[14, 20, 26, 34, 40, 46, 56, 62, 68].map((y, i) => <line key={y} x1={28} y1={y} x2={[104, 96, 84, 106, 92, 70, 100, 88, 76][i]} y2={y} {...m} />)}
    <path d="M14 30 H122 M14 51 H122" stroke={a} strokeDasharray="4 2" />
    <path d="M126 8 v20 M126 32 v17 M126 53 v19" {...ac} />
  </>),
  "document-ingestion": () => (<>
    {[[8, 12], [14, 18], [20, 24]].map(([x, y]) => <path key={x} d={`M${x} ${y} h16 l6 6 v26 h-22 z`} fill={S} stroke={mu} strokeOpacity={0.7} />)}
    <path d="M46 40 H58" {...m} />
    <path d="M60 22 h34 l-12 16 v14 h-10 v-14 z" fill={a} fillOpacity={0.15} stroke={a} />
    <path d="M98 40 H108" {...m} />
    {[46, 38, 30].map((y) => <rect key={y} x={112} y={y} width={22} height={6} rx={1} fill={a} />)}
  </>),
  "parsing-and-extraction": () => (<>
    <rect x={10} y={6} width={60} height={68} rx={2} fill={S} stroke={mu} strokeOpacity={0.6} />
    {[12, 18, 24, 30, 36].flatMap((y) => [[16, 36], [44, 64]].map(([x1, x2]) => <line key={`${y}${x1}`} x1={x1} y1={y} x2={x2} y2={y} {...m} />))}
    <path d="M16 46 h48 v22 h-48 z M16 53 h48 M16 60 h48 M32 46 v22 M48 46 v22" {...m} />
    <path d="M76 40 H84 M81 37 l3 3 l-3 3" {...m} />
    {[18, 28, 38, 58].map((y) => <line key={y} x1={90} y1={y} x2={130} y2={y} stroke={a} strokeWidth={2} />)}
    <line x1={90} y1={48} x2={124} y2={48} {...m} strokeDasharray="2 3" />
  </>),
  "similarity-search": () => (<>
    {dots([[20, 18], [36, 62], [104, 16], [120, 54], [96, 68], [24, 44], [128, 30]], mu, 2.5)}
    <circle cx={66} cy={40} r={20} stroke={a} fill={a} fillOpacity={0.08} strokeDasharray="4 2" />
    {dots([[56, 30], [78, 46], [60, 52]])}<path d="M62 36 l8 8 M70 36 l-8 8" stroke={a} strokeWidth={2} />
  </>),
  "vector-databases": () => (<>
    <path d="M40 14 H100 M24 40 H58 M58 40 H82 M82 40 H116 M14 66 H126" {...m} />
    {dots([[40, 14], [100, 14]], mu)}{dots([[24, 40], [58, 40], [82, 40], [116, 40]], mu)}
    {dots([14, 30, 46, 62, 78, 94, 110, 126].map((x) => [x, 66]), mu, 2)}
    <path d="M100 14 L82 40 L58 40 L62 66" {...ac} />{dots([[100, 14], [82, 40], [58, 40], [62, 66]])}
  </>),
  "hybrid-retrieval": () => (<>
    {[30, 24, 20, 14, 10].map((w, i) => <rect key={i} x={6} y={10 + i * 13} width={w} height={8} rx={1} fill={mf} />)}
    <text x={44} y={44} {...T} fontSize={12}>+</text>
    {[28, 26, 16, 12, 8].map((w, i) => <rect key={i} x={52} y={10 + i * 13} width={w} height={8} rx={1} fill={mf} />)}
    <path d="M86 40 H96 M92 36 l4 4 l-4 4" {...m} />
    {[32, 26, 20, 14, 8].map((w, i) => <rect key={i} x={102} y={10 + i * 13} width={w} height={8} rx={1} fill={a} />)}
  </>),
  "top-k-retrieval": () => (<>
    {bars(10, 13, 9, [52, 46, 40, 32, 26, 20, 16, 12, 10], (i) => (i < 4 ? a : mf), 72)}
    <path d="M10 14 v-5 h48 v5" {...ac} /><text x={34} y={7} {...T} fill={a} fontSize={8}>k</text>
  </>),
  reranking: () => {
    const to = [2, 0, 4, 1, 3];
    return (<>
      {to.map((t, i) => <line key={i} x1={50} y1={14 + i * 13} x2={90} y2={14 + t * 13} stroke={t === 0 ? a : mu} strokeOpacity={t === 0 ? 1 : 0.6} strokeWidth={t === 0 ? 1.5 : 1} />)}
      {[0, 1, 2, 3, 4].map((i) => <rect key={i} x={10} y={10 + i * 13} width={40} height={8} rx={1} fill={mf} />)}
      {[0, 1, 2, 3, 4].map((i) => <rect key={i} x={90} y={10 + i * 13} width={40} height={8} rx={1} fill={i ? mf : a} />)}
    </>);
  },
  "context-assembly": () => (<>
    <rect x={8} y={10} width={124} height={22} rx={3} fill="none" stroke={mu} strokeDasharray="4 2" />
    {boxes([[12, 14, 42, 14], [58, 14, 30, 14], [92, 14, 26, 14]], a, "color-mix(in srgb, var(--accent) 20%, transparent)")}
    {boxes([[20, 50, 34, 14], [70, 50, 26, 14]])}
    {cross(83, 57)}
  </>),
  "prompt-construction": () => (<>
    <rect x={36} y={4} width={68} height={72} rx={3} fill="none" stroke={mu} strokeDasharray="4 2" />
    {boxes([[40, 8, 60, 12], [40, 50, 60, 12]])}{boxes([[40, 24, 60, 22]], a, "color-mix(in srgb, var(--accent) 20%, transparent)")}
    <text x={70} y={17} {...T} fontSize={8}>system</text><text x={70} y={38} {...T} fill={a} fontSize={8}>context</text><text x={70} y={59} {...T} fontSize={8}>question</text>
  </>),
  "rag-pipeline": () => (<>
    {[10, 42, 74, 106].map((x) => <rect key={x} x={x} y={14} width={24} height={14} rx={3} fill={S} stroke={mu} />)}
    {[10, 42, 74, 106].map((x) => <rect key={x} x={x} y={52} width={24} height={14} rx={3} fill={S} stroke={a} />)}
    <path d="M34 21h8M66 21h8M98 21h8M34 59h8M66 59h8M98 59h8M118 28v24" {...m} />
  </>),
  "citation-mapping": () => (<>
    <text x={18} y={45} {...T} fill={a} fontSize={13}>[1]</text>
    {boxes([[38, 32, 20, 16], [68, 32, 20, 16], [98, 32, 20, 16]])}
    <path d="M30 40 H38 M58 40 H68 M88 40 H98 M118 40 H124" stroke={a} strokeDasharray="3 2" />
    <circle cx={128} cy={40} r={4} fill={a} />
  </>),
  "grounded-generation": () => (<>
    {[16, 30, 44, 58].map((y) => <line key={y} x1={8} y1={y} x2={56} y2={y} {...m} />)}
    <rect x={84} y={8} width={48} height={64} rx={2} fill={S} stroke={mu} strokeOpacity={0.6} />
    {[[16, 18], [30, 36], [44, 54]].map(([y, sy]) => (<g key={y}>
      <rect x={90} y={sy - 3} width={34} height={6} rx={1} fill={a} fillOpacity={0.3} />
      <path d={`M58 ${y} L88 ${sy}`} stroke={a} /><circle cx={58} cy={y} r={2.5} fill={a} />
    </g>))}
    <text x={64} y={61} {...T} fontSize={9}>?</text>
  </>),
  "hallucination-analysis": () => (<>
    {boxes([[8, 18, 38, 16], [50, 18, 28, 16], [82, 18, 50, 16]])}
    {check(27, 52)}{check(64, 52)}{cross(107, 51, a)}
  </>),
  "retrieval-failures": () => (<>
    {[8, 18, 28, 38].map((y) => <rect key={y} x={24} y={y} width={80} height={7} rx={1} fill={mf} />)}
    <line x1={14} y1={50} x2={114} y2={50} stroke={a} strokeDasharray="4 2" /><text x={8} y={53} {...T} fill={a} fontSize={8}>k</text>
    <rect x={24} y={58} width={80} height={7} rx={1} fill={a} />
    <path d="M110 62 Q 128 40 110 14 M108 20 l2 -6 l5 4" {...ac} />
  </>),
  "rag-evaluation": () => (<>
    {[0, 1, 2, 3, 4].flatMap((r) => [0, 1, 2, 3, 4, 5].map((c) => {
      const ok = (r * 7 + c * 3) % 5 !== 0;
      return <circle key={`${r}${c}`} cx={22 + c * 19} cy={12 + r * 14} r={4} fill={ok ? a : S} stroke={ok ? a : mu} />;
    }))}
  </>),
  "agent-loop": () => (<>
    <circle cx={70} cy={40} r={24} fill="none" stroke={a} strokeDasharray="5 3" />
    {[[70, 16], [94, 40], [70, 64], [46, 40]].map(([x, y]) => <circle key={`${x}${y}`} cx={x} cy={y} r={5} fill={S} stroke={a} />)}
    <rect x={108} y={30} width={24} height={20} rx={3} fill={S} stroke={mu} />
    <line x1={99} y1={40} x2={108} y2={40} {...m} />
  </>),
  "agent-termination": () => (<>
    <circle cx={26} cy={40} r={16} {...ac} strokeDasharray="5 3" /><path d="M38 28 l4 -2 l-1 5" {...ac} />
    {[14, 30, 46, 62].map((y, i) => (<g key={y}>
      <rect x={56} y={y} width={76} height={8} rx={2} fill="none" stroke={mu} strokeOpacity={0.6} />
      <rect x={56} y={y} width={[40, 76, 30, 52][i]} height={8} rx={2} fill={i === 1 ? a : mf} />
    </g>))}
  </>),
  "tool-calling": () => (<>
    {boxes([[6, 28, 32, 24], [56, 28, 28, 24]])}{boxes([[102, 28, 32, 24]], a)}
    <text x={22} y={43} {...T} fontSize={8}>LLM</text><text x={118} y={44} {...T} fill={a} fontSize={10}>f()</text>
    <path d="M38 35 H56 M84 35 H102 M98 32 l4 3 l-4 3" {...ac} />
    <path d="M102 45 H84 M56 45 H38 M42 42 l-4 3 l4 3" {...m} />
  </>),
  "tool-selection": () => (<>
    <rect x={8} y={32} width={24} height={16} rx={3} fill={a} fillOpacity={0.2} stroke={a} />
    {[13, 31, 49, 67].map((y, i) => <line key={y} x1={32} y1={40} x2={84} y2={y} stroke={i === 1 ? a : mu} strokeOpacity={i === 1 ? 1 : 0.5} strokeWidth={i === 1 ? 1.6 : 1} />)}
    {[8, 26, 44, 62].map((y, i) => <rect key={y} x={84} y={y} width={40} height={10} rx={2} fill={S} stroke={i === 1 ? a : mu} strokeWidth={i === 1 ? 1.5 : 1} />)}
  </>),
  "tool-argument-validation": () => (<>
    <path d="M18 8 V60 M18 16 H30 M18 30 H30 M18 46 H30 M18 60 H30 M40 38 V46 H48" {...m} />
    {boxes([[30, 12, 56, 8], [30, 26, 56, 8], [30, 56, 56, 8]])}
    <rect x={48} y={42} width={46} height={8} rx={2} fill={a} fillOpacity={0.2} stroke={a} />
    {check(100, 15, mu)}{check(100, 29, mu)}{check(100, 59, mu)}{cross(104, 46, a)}
  </>),
  "structured-output-validation": () => (<>
    <text x={22} y={50} {...T} fill={a} fontSize={24}>{"{}"}</text>
    {[56, 80, 104].map((x) => <path key={x} d={`M${x} 10 V34 M${x} 46 V70`} stroke={mu} strokeOpacity={0.7} strokeWidth={3} />)}
    <path d="M42 40 H120" stroke={a} strokeDasharray="3 3" />{check(128, 40)}
  </>),
  "short-term-and-long-term-memory": () => (<>
    <rect x={6} y={10} width={80} height={28} rx={3} fill="none" stroke={a} strokeDasharray="4 2" />
    {boxes([[12, 16, 20, 16], [36, 16, 20, 16], [60, 16, 20, 16]])}
    <rect x={30} y={52} width={20} height={14} rx={2} fill={mf} />
    <path d="M40 40 V50 M54 60 H100" {...m} />
    <path d="M100 52 v18 a14 4 0 0 0 28 0 v-18" fill={S} stroke={a} /><ellipse cx={114} cy={52} rx={14} ry={4} fill={S} stroke={a} />
    <path d="M114 46 Q 112 24 90 24" stroke={a} fill="none" strokeDasharray="3 2" />
  </>),
  "multi-agent-coordination": () => {
    const sat = [[24, 16], [24, 64], [116, 16], [116, 64]];
    return (<>
      {sat.map(([x, y]) => <line key={`${x}${y}`} x1={70} y1={40} x2={x} y2={y} {...m} />)}
      {sat.map(([x, y]) => <circle key={`${x}${y}`} cx={x} cy={y} r={7} fill={S} stroke={mu} />)}
      <circle cx={70} cy={40} r={10} fill={S} stroke={a} strokeWidth={1.5} /><circle cx={93} cy={28} r={3} fill={a} />
    </>);
  },
  "prompt-injection": () => (<>
    <rect x={8} y={12} width={82} height={56} rx={3} fill={S} stroke={mu} strokeOpacity={0.6} />
    {[22, 32, 52, 60].map((y) => <line key={y} x1={14} y1={y} x2={82} y2={y} {...m} />)}
    <rect x={14} y={38} width={68} height={8} rx={1} fill={a} fillOpacity={0.3} stroke={a} />
    <path d="M114 16 l14 6 v12 c0 12 -7 18 -14 22 c-7 -4 -14 -10 -14 -22 v-12 z" fill={a} fillOpacity={0.15} stroke={a} strokeWidth={1.5} />
  </>),
  "model-serving": () => (<>
    {dots([[10, 22], [18, 50], [26, 34], [12, 62], [30, 58]], mu, 2.5)}
    <rect x={42} y={26} width={22} height={28} rx={3} fill="none" stroke={a} strokeDasharray="3 2" />{dots([[53, 33], [53, 40], [53, 47]])}
    <path d="M64 40 L86 23 M64 40 L86 57" {...m} />
    {boxes([[88, 12, 42, 22]], a)}{boxes([[88, 46, 42, 22]])}
  </>),
  "latency-and-throughput": () => (<>
    {bars(10, 8, 6, [8, 22, 36, 40, 32, 24, 17, 12, 8, 6, 4, 3, 2, 2, 1], (i) => (i >= 10 ? a : mf))}
    <line x1={89} y1={8} x2={89} y2={72} stroke={a} strokeDasharray="3 2" /><text x={102} y={14} {...T} fill={a} fontSize={8}>p95</text>
  </>),
  "rate-limits-and-retries": () => (<>
    <path d="M14 18 h44 l-6 46 h-32 z" fill={S} stroke={mu} />
    {dots([[28, 52], [36, 56], [44, 52], [32, 44], [40, 46]])}
    <path d="M60 40 H132" {...m} />{dots([[72, 40], [84, 40], [104, 40], [130, 40]], a, 3)}
  </>),
  "retries-and-timeouts": () => (<>
    <line x1={6} y1={46} x2={134} y2={46} {...m} />
    {[[8, 14], [36, 14], [74, 14]].map(([x, w]) => <g key={x}><rect x={x} y={34} width={w} height={10} rx={2} fill={mf} />{cross(x + w + 6, 39)}</g>)}
    <rect x={110} y={34} width={12} height={10} rx={2} fill={a} />
    <line x1={128} y1={20} x2={128} y2={62} stroke={a} strokeDasharray="3 2" />
  </>),
  "cost-optimization": () => (<>
    <rect x={10} y={14} width={18} height={56} rx={1} fill={mf} />
    {[[34, 14, 12], [56, 26, 12], [78, 38, 8]].map(([x, y, h]) => <rect key={x} x={x} y={y} width={18} height={h} rx={1} fill="none" stroke={mu} strokeDasharray="2 2" />)}
    <path d="M28 14 H34 M52 26 H56 M74 38 H78 M96 46 H104" {...m} />
    <rect x={104} y={46} width={18} height={24} rx={1} fill={a} />
  </>),
  "distributed-tracing": () => (<>
    {[[10, 10, 120], [18, 22, 60], [82, 22, 44], [26, 34, 30], [86, 46, 36], [32, 58, 16]].map(([x, y, w], i) => <rect key={i} x={x} y={y} width={w} height={8} rx={2} fill={i === 4 ? a : mf} />)}
  </>),
  "monitoring-and-alerting": () => (<>
    <rect x={72} y={8} width={30} height={64} fill={a} fillOpacity={0.1} />
    <line x1={8} y1={30} x2={132} y2={30} stroke={a} strokeDasharray="4 2" />
    <path d="M8 56 L22 50 L36 54 L50 46 L64 40 L76 22 L88 18 L100 26 L112 44 L124 48 L132 52" {...m} strokeOpacity={0.9} strokeWidth={1.5} />
  </>),
  "data-drift-and-concept-drift": () => (<>
    <path d={curve(gauss(48, 13, 50))} {...m} strokeDasharray="3 2" />
    <path d={curve(gauss(86, 13, 50))} {...ac} />
    <path d="M50 12 H80 M76 8 l4 4 l-4 4" {...ac} />
  </>),
  "model-drift": () => (<>
    {bars(10, 10, 7, [6, 7, 5, 8, 9, 12, 16, 24, 32, 38, 10, 7], (i) => (i >= 7 && i <= 9 ? a : mf))}
    <line x1={8} y1={50} x2={132} y2={50} stroke={a} strokeDasharray="4 2" />
    <line x1={107} y1={8} x2={107} y2={72} {...m} strokeDasharray="2 2" />
  </>),
  "feature-and-training-serving-skew": () => (<>
    <text x={24} y={27} {...T} fontSize={8}>batch</text><text x={24} y={55} {...T} fontSize={8}>online</text>
    <rect x={46} y={19} width={82} height={10} rx={2} fill={mf} />
    <rect x={46} y={47} width={60} height={10} rx={2} fill={a} />
    <rect x={106} y={47} width={22} height={10} rx={2} fill="none" stroke={a} strokeDasharray="2 2" />
  </>),
  "model-versioning": () => (<>
    <line x1={10} y1={50} x2={130} y2={50} {...m} />
    {[22, 52, 82, 112].map((x) => <circle key={x} cx={x} cy={50} r={6} fill={S} stroke={x === 82 ? a : mu} strokeWidth={x === 82 ? 1.5 : 1} />)}
    <path d="M82 44 V16 h28 l-5 5 l5 5 h-28" fill={a} fillOpacity={0.2} stroke={a} /><text x={94} y={24} {...T} fill={a} fontSize={7}>prod</text>
  </>),
  "experiment-tracking": () => (<>
    <path d="M12 8 V70 H132" {...m} />
    {[[48, 20], [54, 30], [44, 14]].map(([f, k]) => <path key={f} d={curve((x) => f - (f - 14) * Math.exp(-(x - 14) / k), 14, 130)} {...m} />)}
    <path d={curve((x) => 62 - 48 * Math.exp(-(x - 14) / 25), 14, 130)} {...ac} />
  </>),
  "ci-cd-for-ml": () => (<>
    <line x1={14} y1={40} x2={128} y2={40} {...m} />
    {[14, 40, 66].map((x) => <g key={x}><circle cx={x} cy={40} r={8} fill={a} fillOpacity={0.2} stroke={a} />{check(x, 40)}</g>)}
    <path d="M94 30 l10 10 l-10 10 l-10 -10 z" fill={S} stroke={a} strokeWidth={1.5} />
    <rect x={116} y={32} width={18} height={16} rx={3} fill={S} stroke={mu} />
  </>),
  "golden-datasets": () => (<>
    {[[24, 26], [30, 20], [36, 14]].map(([x, y]) => <rect key={x} x={x} y={y} width={56} height={44} rx={3} fill={S} stroke={mu} />)}
    {[24, 34, 44].map((y) => <g key={y}><line x1={42} y1={y} x2={76} y2={y} {...m} />{check(84, y - 1)}</g>)}
    <polygon points={Array.from({ length: 10 }, (_, i) => { const r = i % 2 ? 6 : 14, t = (i * Math.PI) / 5 - Math.PI / 2; return `${112 + r * Math.cos(t)},${44 + r * Math.sin(t)}`; }).join(" ")} fill={a} fillOpacity={0.3} stroke={a} />
  </>),
  "offline-evaluation": () => (<>
    {[[14, 40, 46], [54, 36, 42], [94, 38, 22]].map(([x, A, B]) => (<g key={x}>
      <rect x={x} y={70 - A} width={12} height={A} rx={1} fill={mf} />
      <rect x={x + 14} y={70 - B} width={12} height={B} rx={1} fill={a} />
      <path d={`M${x + 20} ${64 - B} v12 M${x + 17} ${64 - B} h6 M${x + 17} ${76 - B} h6`} stroke={mu} />
    </g>))}
  </>),
  "online-evaluation": () => (<>
    <path d="M10 10 C 50 22, 90 30, 130 33 L130 45 C 90 48, 50 58, 10 70 z" fill={a} fillOpacity={0.12} />
    <path d="M10 40 C 50 40, 90 39, 130 39" {...ac} />
    <line x1={10} y1={54} x2={130} y2={54} {...m} strokeDasharray="3 3" />
  </>),
  "safety-evaluations": () => (<>
    <rect x={84} y={14} width={48} height={50} fill={a} fillOpacity={0.1} />
    <line x1={8} y1={40} x2={132} y2={40} {...m} />
    {dots([[14, 40], [26, 40], [38, 40], [50, 40], [66, 40], [92, 40]], mu, 3)}{dots([[58, 40], [100, 40], [112, 40], [124, 40]], a, 3)}
    <line x1={84} y1={14} x2={84} y2={64} stroke={a} strokeWidth={1.5} /><circle cx={84} cy={66} r={4} fill={S} stroke={a} strokeWidth={1.5} />
  </>),
};
