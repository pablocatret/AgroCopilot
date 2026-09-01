import type { AgentRunView } from "../hooks/useChatSession"
import type { FinalAnswer } from "../types"

const svgDataUri = (svg: string) => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`

const sceneSvg = ({
  label,
  subtitle,
  palette,
  stress = false,
  water = false,
  radar = false,
  cover = false,
}: {
  label: string
  subtitle: string
  palette: [string, string, string]
  stress?: boolean
  water?: boolean
  radar?: boolean
  cover?: boolean
}) => svgDataUri(`
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 560" role="img" aria-label="${label}">
  <defs>
    <linearGradient id="base" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="${palette[0]}"/>
      <stop offset=".52" stop-color="${palette[1]}"/>
      <stop offset="1" stop-color="${palette[2]}"/>
    </linearGradient>
    <linearGradient id="hot" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#1f7a55"/>
      <stop offset=".48" stop-color="#e7c15d"/>
      <stop offset="1" stop-color="#b95837"/>
    </linearGradient>
    <filter id="softNoise">
      <feTurbulence type="fractalNoise" baseFrequency=".72" numOctaves="2" seed="7"/>
      <feColorMatrix type="saturate" values=".18"/>
      <feBlend mode="soft-light" in2="SourceGraphic"/>
    </filter>
  </defs>
  <rect width="960" height="560" fill="${radar ? "#101418" : "#eef1e4"}"/>
  <g transform="rotate(-7 480 280)" filter="${radar ? "url(#softNoise)" : "none"}">
    <rect x="70" y="58" width="820" height="438" rx="28" fill="url(#base)"/>
    <path d="M98 124 C254 86, 430 118, 590 88 C735 61, 830 92, 872 142 L872 208 C700 188, 548 206, 386 184 C235 164, 148 184, 98 174 Z" fill="${cover ? "#2f7b48" : radar ? "#6f6783" : "#27764a"}" opacity=".78"/>
    <path d="M92 274 C224 236, 360 262, 502 242 C656 220, 770 240, 878 300 L878 380 C706 334, 570 332, 414 346 C270 360, 168 346, 92 362 Z" fill="${stress ? "url(#hot)" : water ? "#1f8096" : radar ? "#e1a15e" : "#7aa653"}" opacity=".84"/>
    <path d="M118 418 C260 392, 430 424, 575 398 C706 374, 802 386, 862 422 L862 474 L118 474 Z" fill="${cover ? "#d6bb63" : stress ? "#679b4b" : "#8ab267"}" opacity=".76"/>
    ${cover ? `<path d="M690 82 L870 96 L870 172 C810 142, 755 132, 690 152 Z" fill="#a74337" opacity=".78"/>` : ""}
    <g stroke="#fff8df" stroke-width="7" opacity="${radar ? ".24" : ".58"}">
      <path d="M162 75 L132 500"/>
      <path d="M310 68 L286 505"/>
      <path d="M458 62 L442 508"/>
      <path d="M606 64 L610 508"/>
      <path d="M748 78 L780 496"/>
      <path d="M852 112 L898 462"/>
    </g>
    <g fill="none" stroke="#172219" stroke-width="2" opacity=".22">
      <path d="M90 86 L884 86 L884 492 L90 492 Z"/>
      <path d="M92 288 L884 288"/>
      <path d="M482 66 L482 506"/>
    </g>
  </g>
  <g transform="translate(36 34)">
    <rect width="278" height="82" rx="16" fill="#07110d" opacity=".78"/>
    <text x="18" y="34" fill="#edf6e8" font-family="Arial, sans-serif" font-size="18" font-weight="700">${label}</text>
    <text x="18" y="60" fill="#b9c7b5" font-family="Arial, sans-serif" font-size="13">${subtitle}</text>
  </g>
  <g transform="translate(716 36)">
    <rect width="190" height="54" rx="14" fill="#ffffff" opacity=".74"/>
    <circle cx="28" cy="27" r="8" fill="${palette[0]}"/>
    <circle cx="62" cy="27" r="8" fill="${palette[1]}"/>
    <circle cx="96" cy="27" r="8" fill="${palette[2]}"/>
    <text x="124" y="32" fill="#334238" font-family="Arial, sans-serif" font-size="12" font-weight="700">DEMO</text>
  </g>
</svg>
`)

const ndviPreview = sceneSvg({
  label: "NDVI actual",
  subtitle: "Recinto OL-17 Norte",
  palette: ["#1c5d42", "#7eaa56", "#d5bd69"],
  stress: true,
})

const previousPreview = sceneSvg({
  label: "NDVI referencia",
  subtitle: "Escena base de marzo",
  palette: ["#246a45", "#7fa95a", "#bed07b"],
})

const changePreview = sceneSvg({
  label: "Delta NDVI",
  subtitle: "Actual menos referencia",
  palette: ["#1d7b59", "#e6c35d", "#b85838"],
  stress: true,
})

const ndwiPreview = sceneSvg({
  label: "NDWI actual",
  subtitle: "Agua superficial",
  palette: ["#e3efe2", "#86b795", "#287f96"],
  water: true,
})

const ndmiPreview = sceneSvg({
  label: "NDMI actual",
  subtitle: "Humedad de canopia",
  palette: ["#1f665e", "#79a96a", "#c27b45"],
  stress: true,
})

const radarPreview = sceneSvg({
  label: "Sentinel-1 VV",
  subtitle: "Backscatter auxiliar",
  palette: ["#11151a", "#5c6074", "#e0a15d"],
  radar: true,
})

const worldcoverPreview = sceneSvg({
  label: "WorldCover",
  subtitle: "Contexto de cobertura",
  palette: ["#2d7b47", "#d3ba61", "#3f89a5"],
  cover: true,
})

export const productDemoQuery =
  "Explotacion ficticia Las Lomas, olivar superintensivo en Jaen. En el recinto OL-17 Norte aparece una perdida de vigor en dos calles desde finales de marzo. Revisar NDVI, NDWI, NDMI, Sentinel-1 y cobertura; contrastar con cuaderno de riego, justificantes de ayuda y proponer que comprobar antes de decidir una actuacion."

export const productDemoRuns: AgentRunView[] = [
  { key: "organizer#1", agent: "organizer", runId: 1, totalRuns: 1, status: "done", attempt: 1, attemptLimit: 1, executionLevel: "ok", detail: "Ruta multiagente seleccionada: STAC, legal, documental y continuidad" },
  { key: "stac#1", agent: "stac", runId: 1, totalRuns: 1, status: "done", attempt: 1, attemptLimit: 1, executionLevel: "ok", detail: "Escenas Sentinel-2, Sentinel-1 y WorldCover normalizadas" },
  { key: "rs_analyst#1", agent: "rs_analyst", runId: 1, totalRuns: 1, status: "done", attempt: 1, attemptLimit: 1, executionLevel: "insufficient_data", detail: "Patron multisenal consistente; causalidad pendiente de campo" },
  { key: "legal#1", agent: "legal", runId: 1, totalRuns: 1, status: "done", attempt: 1, attemptLimit: 1, executionLevel: "ok", detail: "Requisitos y cautelas normativas estructurados" },
  { key: "document_analyst#1", agent: "document_analyst", runId: 1, totalRuns: 1, status: "done", attempt: 1, attemptLimit: 1, executionLevel: "ok", detail: "Adjuntos y soporte documental resumidos" },
  { key: "case_manager#1", agent: "case_manager", runId: 1, totalRuns: 1, status: "done", attempt: 1, attemptLimit: 1, executionLevel: "ok", detail: "Tareas, bloqueos y siguiente input actualizados" },
  { key: "writer#1", agent: "writer", runId: 1, totalRuns: 1, status: "done", attempt: 1, attemptLimit: 1, executionLevel: "ok", detail: "Respuesta final sintetizada con trazabilidad" },
]

export const productDemoAnswer: FinalAnswer = {
  language: "es",
  executive_summary:
    "La evidencia apunta a una incidencia localizada en el sector norte del recinto OL-17: NDVI y NDMI caen de forma coherente, NDWI no muestra agua superficial persistente y Sentinel-1 solo refuerza la necesidad de comprobar humedad/rugosidad en campo. La decision recomendada es inspeccionar riego y suelo antes de actuar, mientras se completa el soporte documental para no confundir una senal satelital con una justificacion cerrada.",
  next_actions: [
    "Inspeccionar hoy el ramal norte: presion, filtros, goteros, humedad a 20-40 cm y sintomas foliares en las dos calles afectadas.",
    "Comparar la zona afectada con una calle sana del mismo sector y registrar fotos georreferenciadas con fecha y punto de muestreo.",
    "Preparar la actuacion solo como decision provisional hasta cerrar cuaderno de riego, croquis SIGPAC y justificante tecnico.",
  ],
  missing_information: [
    "Lectura de presion o caudal del ramal norte durante el ultimo riego.",
    "Fecha exacta de la ultima fertirrigacion y cualquier incidencia de filtrado.",
    "Confirmacion en campo de suelo somero, compactacion o posible obstruccion localizada.",
  ],
  documents_needed: [
    "Croquis SIGPAC del recinto OL-17 con delimitacion de la subzona norte.",
    "Cuaderno de explotacion actualizado con riegos y fertirrigaciones de marzo-abril.",
    "Fotos georreferenciadas y nota tecnica breve antes de justificar una actuacion.",
  ],
  evidence_summary: [
    "NDVI actual 0.531 frente a 0.617 en marzo: bajada media de -0.086 concentrada en el norte.",
    "NDMI cae -0.082 y refuerza la hipotesis de menor humedad/canopia; NDWI no confirma lamina de agua persistente.",
    "Sentinel-1 VV se usa como senal auxiliar porque puede mezclar humedad superficial, rugosidad y geometria SAR.",
    "El panel documental distingue requisitos verificados, pendientes y con soporte insuficiente antes de presentar nada.",
  ],
  legal: {
    answer:
      "La base documental permite avanzar la preparacion del expediente, pero no cerrar cumplimiento: la titularidad y el recinto estan identificados, el cuaderno de explotacion requiere actualizacion y la senal satelital solo sirve como apoyo tecnico si se acompana de verificacion de campo.",
    jurisdiction: "ES/Andalucia",
    source_status: "vigente a contrastar",
    updated_at: "2026-04-15",
    official_source: "https://www.boe.es/",
    limitations: [
      "La fuente oficial debe contrastarse antes de presentar una solicitud real.",
      "La teledeteccion no acredita por si sola causa agronomica ni cumplimiento documental.",
      "El caso demo usa datos ficticios no sensibles.",
    ],
    checklist: [
      {
        requirement: "Identificacion de recinto, cultivo y titularidad declarada",
        status: "cumple",
        evidence: [{ title: "Ficha interna demo OL-17", source: "demo" }],
        jurisdiction: "ES/Andalucia",
        source_status: "declarado",
        updated_at: "2026-04-15",
        official_source: "https://www.boe.es/",
      },
      {
        requirement: "Cuaderno de explotacion con riegos y fertirrigacion recientes",
        status: "no_cumple",
        evidence: [{ title: "Extracto cuaderno riego marzo-abril", source: "demo" }],
        jurisdiction: "Andalucia",
        source_status: "pendiente",
        updated_at: "2026-04-15",
        limitations: ["Falta el registro de presion/caudal del ramal norte y cierre de actuaciones de abril."],
      },
      {
        requirement: "Justificacion tecnica de la actuacion correctiva",
        status: "insuficiente",
        evidence: [{ title: "Informe multisenal Sentinel demo", source: "demo" }],
        jurisdiction: "ES",
        source_status: "requiere contraste",
        updated_at: "2026-04-15",
        limitations: ["NDVI/NDMI orientan la inspeccion, pero no demuestran causalidad sin observacion de campo."],
      },
      {
        requirement: "Evidencia grafica fechada y georreferenciada",
        status: "no_cumple",
        evidence: [{ title: "Checklist documental demo", source: "demo" }],
        jurisdiction: "Andalucia",
        source_status: "pendiente",
        updated_at: "2026-04-15",
        limitations: ["Conviene registrar imagenes de la zona afectada y zona control antes de ejecutar la actuacion."],
      },
    ],
  },
  remote_sensing: {
    overview:
      "Lectura multisenal para OL-17 Norte: vigor y humedad de canopia bajan de forma localizada, agua superficial no queda confirmada y radar/WorldCover se mantienen como contexto auxiliar.",
    insights: [
      {
        item_id: "S2A_MSIL2A_20260412_DEMO_OL17::NDVI",
        summary: "NDVI recortado sobre OL-17: media 0.531 con 12 840 pixeles validos; la anomalia se concentra en dos calles del sector norte.",
        confidence: 0.78,
        product_label: "NDVI recortado",
        stats: { index_name: "NDVI", mean: 0.531, min: 0.18, max: 0.78, std: 0.091, valid_pixels: 12840, masked_pixels: 820, quality_mask_applied: true },
        quality: { label: "media", cloud_cover: 0.08, reasons: ["Mascara SCL aplicada", "Borde de parcela aproximado"] },
        limitations: ["NDVI debe leerse con fenologia, variedad, marco de plantacion y calidad de escena."],
      },
      {
        item_id: "S2A_MSIL2A_20260412_DEMO_OL17::NDWI",
        summary: "NDWI medio -0.042; no aparece una lamina de agua persistente ni encharcamiento claro.",
        confidence: 0.68,
        product_label: "NDWI recortado",
        stats: { index_name: "NDWI", mean: -0.042, min: -0.31, max: 0.19, std: 0.057, valid_pixels: 12790, masked_pixels: 870, quality_mask_applied: true },
        quality: { label: "media", cloud_cover: 0.08, reasons: ["Mascara de calidad aplicada"] },
        limitations: ["NDWI senala agua o humedad superficial visible; no confirma por si solo el estado del riego."],
      },
      {
        item_id: "S2A_MSIL2A_20260412_DEMO_OL17::NDMI",
        summary: "NDMI medio 0.246; la lectura refuerza menor humedad/canopia en la franja norte.",
        confidence: 0.73,
        product_label: "NDMI recortado",
        stats: { index_name: "NDMI", mean: 0.246, min: -0.08, max: 0.49, std: 0.074, valid_pixels: 12680, masked_pixels: 980, quality_mask_applied: true },
        quality: { label: "media", cloud_cover: 0.08, reasons: ["SWIR disponible", "Parcela recortada con margen tecnico"] },
        limitations: ["NDMI orienta sobre humedad relativa de vegetacion/canopia; no diagnostica estres hidrico por si solo."],
      },
      {
        item_id: "S1A_RTC_20260413_DEMO_OL17",
        summary: "Radar Sentinel-1 VV medio -11.8 dB; evidencia auxiliar compatible con cambios superficiales, rugosidad o laboreo.",
        confidence: 0.6,
        product_label: "Radar Sentinel-1 VV recortado",
        stats: { index_name: "S1_VV", mean: -11.8, min: -19.4, max: -4.6, std: 2.4, valid_pixels: 9800, masked_pixels: 0 },
        quality: { label: "desconocida", reasons: ["Sin mascara de nubes", "Interpretacion condicionada por geometria SAR"] },
        limitations: ["Sentinel-1 es sensible a humedad superficial, rugosidad, laboreo, orientacion y geometria de adquisicion."],
      },
      {
        item_id: "ESA_WORLDCOVER_2021_DEMO_OL17",
        summary: "WorldCover recortado: cultivo 81.4%, arbolado 11.6% y camino/suelo desnudo 4.2%; sirve como contexto, no como diagnostico temporal.",
        confidence: 0.64,
        product_label: "ESA WorldCover recortado",
        stats: {
          index_name: "ESA_WORLDCOVER",
          valid_pixels: 14200,
          masked_pixels: 0,
          class_stats: [
            { code: 40, label: "cultivo", pixels: 11559, percent: 81.4 },
            { code: 10, label: "arbolado", pixels: 1647, percent: 11.6 },
            { code: 60, label: "suelo/camino", pixels: 596, percent: 4.2 },
          ],
        },
        quality: { label: "desconocida", reasons: ["Producto global de cobertura del suelo"] },
        limitations: ["WorldCover no sustituye SIGPAC, catastro ni verificacion de campo."],
      },
    ],
    temporal_changes: [
      {
        from_item_id: "S2B_MSIL2A_20260318_DEMO_OL17",
        to_item_id: "S2A_MSIL2A_20260412_DEMO_OL17",
        label: "Descenso localizado de vigor",
        detail:
          "El delta medio es negativo y la franja norte concentra la perdida, un patron compatible con incidencia de riego o suelo que debe validarse sobre el terreno.",
        confidence: 0.76,
        metric: "NDVI",
        delta_mean: -0.086,
        severity: "media",
        reliable: true,
        limitations: ["Resolucion Sentinel-2 insuficiente para diagnostico de arbol individual.", "Fenologia y suelo visible pueden explicar parte del cambio."],
        preview_href: changePreview,
      },
      {
        from_item_id: "S2B_MSIL2A_20260318_DEMO_OL17::NDWI",
        to_item_id: "S2A_MSIL2A_20260412_DEMO_OL17::NDWI",
        label: "NDWI sin agua persistente",
        detail: "NDWI pasa de -0.018 a -0.042. La variacion no confirma encharcamiento; se mantiene como contexto de humedad superficial.",
        confidence: 0.66,
        metric: "NDWI",
        delta_mean: -0.024,
        severity: "baja",
        reliable: false,
        limitations: ["NDWI no equivale a riego confirmado.", "La lluvia reciente o suelo desnudo pueden alterar la lectura."],
      },
      {
        from_item_id: "S2B_MSIL2A_20260318_DEMO_OL17::NDMI",
        to_item_id: "S2A_MSIL2A_20260412_DEMO_OL17::NDMI",
        label: "Descenso de NDMI",
        detail: "NDMI pasa de 0.328 a 0.246; senal compatible con menor humedad relativa de vegetacion/canopia.",
        confidence: 0.73,
        metric: "NDMI",
        delta_mean: -0.082,
        severity: "media",
        reliable: true,
        limitations: ["NDMI mezcla canopia, fenologia y suelo visible; requiere contraste de riego y suelo."],
      },
    ],
    focus_areas: [
      { title: "Ramal norte", detail: "Comprobar presion, obstrucciones, uniformidad y humedad a 20-40 cm.", parcel: "OL-17 Norte", priority: "alta" },
      { title: "Calle oeste", detail: "Contrastar si el patron coincide con suelo somero, compactacion o pendiente.", parcel: "OL-17 Oeste", priority: "media" },
      { title: "Zona control", detail: "Tomar referencia en una calle sana del mismo sector para comparar vigor y humedad.", parcel: "OL-17 Centro", priority: "media" },
    ],
  },
  temporal_comparison: {
    available: true,
    label: "Comparacion temporal Sentinel-2",
    rationale: "Escenas separadas por 25 dias, con nubosidad baja y mascara de calidad aplicada antes de calcular indices.",
    metric: "NDVI",
    delta_mean: -0.086,
    severity: "media",
    confidence: 0.76,
    key_changes: [
      "Descenso de vigor concentrado en el sector norte frente a la referencia de marzo.",
      "La zona central permanece estable y actua como contraste interno dentro del recinto.",
      "NDMI refuerza la hipotesis de humedad/canopia reducida; NDWI no confirma agua superficial persistente.",
    ],
    limitations: ["Producto demo no oficial; no representa una delimitacion catastral.", "No se infiere causa agronomica sin datos de campo."],
    previous: {
      item_id: "S2B_MSIL2A_20260318_DEMO_OL17::NDVI",
      datetime: "2026-03-18T10:45:00Z",
      preview_href: previousPreview,
      product_label: "NDVI recortado",
      summary: "Vigor alto y homogeneo antes del episodio, sin franja norte destacada.",
      stats: { index_name: "NDVI", mean: 0.617, min: 0.22, max: 0.82, std: 0.073, valid_pixels: 13120 },
      quality: { label: "alta", cloud_cover: 0.03 },
    },
    current: {
      item_id: "S2A_MSIL2A_20260412_DEMO_OL17::NDVI",
      datetime: "2026-04-12T10:48:00Z",
      preview_href: ndviPreview,
      product_label: "NDVI recortado",
      summary: "Bajada localizada en el norte y borde oeste; resto de la parcela estable.",
      stats: { index_name: "NDVI", mean: 0.531, min: 0.18, max: 0.78, std: 0.091, valid_pixels: 12840 },
      quality: { label: "media", cloud_cover: 0.08 },
    },
    change_preview_href: changePreview,
  },
  stac: {
    temporal_selection: {
      previous_item_id: "S2B_MSIL2A_20260318_DEMO_OL17::NDVI",
      current_item_id: "S2A_MSIL2A_20260412_DEMO_OL17::NDVI",
      rationale: "Par de escenas con separacion suficiente, baja nubosidad y cobertura completa del recinto.",
      strategy: "quality_and_gap",
      preferred_min_gap_days: 21,
      actual_gap_days: 25,
      used_multi_window_search: true,
      query_windows: ["2026-03-12/2026-03-25", "2026-04-08/2026-04-16"],
    },
    items: [
      {
        id: "S2A_MSIL2A_20260412_DEMO_OL17::NDVI",
        datetime: "2026-04-12T10:48:00Z",
        collection: "sentinel-2-l2a",
        product_type: "spectral_index",
        product_label: "NDVI recortado",
        index_name: "NDVI",
        index_stats: { index_name: "NDVI", mean: 0.531, min: 0.18, max: 0.78, std: 0.091, valid_pixels: 12840, masked_pixels: 820, quality_mask_applied: true },
        quality: { label: "media", cloud_cover: 0.08 },
        assets: [{ href: ndviPreview, thumbnail: ndviPreview, title: "NDVI preview", mime_type: "image/svg+xml" }],
      },
      {
        id: "S2B_MSIL2A_20260318_DEMO_OL17::NDVI",
        datetime: "2026-03-18T10:45:00Z",
        collection: "sentinel-2-l2a",
        product_type: "spectral_index",
        product_label: "NDVI referencia",
        index_name: "NDVI",
        index_stats: { index_name: "NDVI", mean: 0.617, min: 0.22, max: 0.82, std: 0.073, valid_pixels: 13120, quality_mask_applied: true },
        quality: { label: "alta", cloud_cover: 0.03 },
        assets: [{ href: previousPreview, thumbnail: previousPreview, title: "Reference NDVI", mime_type: "image/svg+xml" }],
      },
      {
        id: "S2A_MSIL2A_20260412_DEMO_OL17::NDWI",
        datetime: "2026-04-12T10:48:00Z",
        collection: "sentinel-2-l2a",
        product_type: "spectral_index",
        product_label: "NDWI recortado",
        index_name: "NDWI",
        index_stats: { index_name: "NDWI", mean: -0.042, min: -0.31, max: 0.19, std: 0.057, valid_pixels: 12790, masked_pixels: 870, quality_mask_applied: true },
        quality: { label: "media", cloud_cover: 0.08 },
        assets: [{ href: ndwiPreview, thumbnail: ndwiPreview, title: "NDWI preview", mime_type: "image/svg+xml" }],
      },
      {
        id: "S2A_MSIL2A_20260412_DEMO_OL17::NDMI",
        datetime: "2026-04-12T10:48:00Z",
        collection: "sentinel-2-l2a",
        product_type: "spectral_index",
        product_label: "NDMI recortado",
        index_name: "NDMI",
        index_stats: { index_name: "NDMI", mean: 0.246, min: -0.08, max: 0.49, std: 0.074, valid_pixels: 12680, masked_pixels: 980, quality_mask_applied: true },
        quality: { label: "media", cloud_cover: 0.08 },
        assets: [{ href: ndmiPreview, thumbnail: ndmiPreview, title: "NDMI preview", mime_type: "image/svg+xml" }],
      },
      {
        id: "S1A_RTC_20260413_DEMO_OL17",
        datetime: "2026-04-13T06:12:00Z",
        collection: "sentinel-1-rtc",
        product_type: "radar",
        product_label: "Radar Sentinel-1 VV",
        index_name: "S1_VV",
        index_stats: { index_name: "S1_VV", mean: -11.8, min: -19.4, max: -4.6, std: 2.4, valid_pixels: 9800, masked_pixels: 0 },
        quality: { label: "desconocida" },
        assets: [{ href: radarPreview, thumbnail: radarPreview, title: "Sentinel-1 VV", mime_type: "image/svg+xml" }],
      },
      {
        id: "ESA_WORLDCOVER_2021_DEMO_OL17",
        datetime: "2021-01-01T00:00:00Z",
        collection: "esa-worldcover",
        product_type: "landcover",
        product_label: "ESA WorldCover",
        index_name: "ESA_WORLDCOVER",
        index_stats: {
          index_name: "ESA_WORLDCOVER",
          valid_pixels: 14200,
          class_stats: [
            { code: 40, label: "cultivo", pixels: 11559, percent: 81.4 },
            { code: 10, label: "arbolado", pixels: 1647, percent: 11.6 },
            { code: 60, label: "suelo/camino", pixels: 596, percent: 4.2 },
          ],
        },
        quality: { label: "desconocida" },
        assets: [{ href: worldcoverPreview, thumbnail: worldcoverPreview, title: "WorldCover preview", mime_type: "image/svg+xml" }],
      },
    ],
  },
  case_state: {
    case_summary:
      "OL-17 Norte queda abierto como expediente operativo: senal satelital consistente, causa no cerrada, documentacion parcialmente preparada y decision condicionada a inspeccion de riego/campo.",
    open_tasks: [
      { title: "Verificar ramal norte en campo", priority: "alta", status: "abierta", rationale: "NDVI/NDMI apuntan a una incidencia localizada, pero falta confirmar riego y suelo.", source: "remote_sensing" },
      { title: "Actualizar cuaderno de riego", priority: "alta", status: "bloqueada", rationale: "La decision documental no puede cerrarse sin presion/caudal y fecha de ultima fertirrigacion.", source: "general" },
      { title: "Adjuntar fotos georreferenciadas", priority: "media", status: "abierta", rationale: "Permiten contrastar satelite con sintomas reales y zona control.", source: "general" },
      { title: "Revisar cautelas legales", priority: "media", status: "abierta", rationale: "La actuacion puede prepararse, pero no presentarse como cumplimiento cerrado.", source: "general" },
    ],
    blocked_by: [
      "Falta registro de presion/caudal del ramal norte.",
      "No hay fotos georreferenciadas de zona afectada y zona control.",
      "Croquis SIGPAC pendiente de exportacion oficial.",
    ],
    recommended_next_input: [
      "Subir cuaderno de riego actualizado o indicar presion/caudal del ultimo riego.",
      "Adjuntar 3-5 fotos georreferenciadas de OL-17 Norte y una zona sana.",
      "Confirmar si la franja afectada coincide con suelo somero, pendiente o compactacion.",
    ],
  },
  cost_summary: {
    conversation_id: "demo-product-ol17",
    total_cost_usd: 0.0847,
    total_tokens: 42680,
    input_tokens: 29140,
    cached_input_tokens: 6420,
    output_tokens: 13540,
    web_calls: 3,
    estimated: true,
    event_count: 12,
    top_model: "gpt-demo-orchestrated",
    top_model_cost_usd: 0.0462,
    warning: false,
    warning_threshold_usd: 0.25,
    by_model: {},
    by_agent: {},
    by_operation: {},
    events: [],
  },
  execution: {
    organizer: { final_level: "ok", instances: [{ instance_id: 1, level: "ok", message: "Ruta multiagente preparada" }] },
    stac: { final_level: "ok", instances: [{ instance_id: 1, level: "ok", message: "Escenas STAC demo normalizadas" }] },
    legal: { final_level: "ok", instances: [{ instance_id: 1, level: "ok", message: "Checklist legal estructurado" }] },
    rs_analyst: { final_level: "insufficient_data", instances: [{ instance_id: 1, level: "insufficient_data", message: "Senal consistente; causalidad pendiente de campo" }] },
    document_analyst: { final_level: "ok", instances: [{ instance_id: 1, level: "ok", message: "Adjuntos resumidos" }] },
    case_manager: { final_level: "ok", instances: [{ instance_id: 1, level: "ok", message: "Continuidad actualizada" }] },
  },
  memory: {
    enabled: true,
    user_id: "demo-las-lomas",
    used_sections: ["profile", "farm_context", "open_questions"],
  },
  attachments: [
    { attachment_id: "att-demo-1", filename: "cuaderno-riego-mar-abr.xlsx", content_type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", size_bytes: 184320, summary: "Registro parcial de riegos y fertirrigacion." },
    { attachment_id: "att-demo-2", filename: "croquis-ol17-borrador.png", content_type: "image/png", size_bytes: 96240, summary: "Delimitacion interna de la subzona norte." },
    { attachment_id: "att-demo-3", filename: "ficha-recinto-ol17.pdf", content_type: "application/pdf", size_bytes: 231424, summary: "Ficha tecnica ficticia de recinto y cultivo." },
  ],
  references: [
    { ref_id: "demo-rs-1", title: "Sentinel-2 L2A demo OL-17", source: "stac", snippet: "Escenas ficticias normalizadas para mostrar seleccion temporal, indices espectrales y calidad de escena sin usar datos sensibles." },
    { ref_id: "demo-rs-2", title: "Sentinel-1 RTC demo OL-17", source: "stac", snippet: "Backscatter VV utilizado solo como evidencia auxiliar por su sensibilidad a humedad superficial, rugosidad y geometria SAR." },
    { ref_id: "demo-doc-1", title: "Cuaderno de riego marzo-abril", source: "document", snippet: "El registro contiene fechas generales de riego, pero no presion/caudal del ramal norte ni cierre de la incidencia." },
    { ref_id: "demo-legal-1", title: "Fuente normativa oficial pendiente de contraste", source: "legal", url: "https://www.boe.es/", snippet: "Referencia publica usada como destino oficial de contraste; la demo no presupone cumplimiento normativo cerrado." },
  ],
  recommendations: [
    "No ejecutar una actuacion correctiva costosa hasta confirmar hidraulica y suelo en campo.",
    "Usar la comparacion multisenal como priorizacion de inspeccion, no como diagnostico causal.",
    "Cerrar el paquete documental antes de convertir la decision tecnica en expediente presentable.",
  ],
  limitations: [
    "Caso demo ficticio y no sensible; no sustituye una evaluacion agronomica real.",
    "La delimitacion del recinto y los productos satelitales son deterministas para documentacion.",
    "La evidencia satelital orienta la inspeccion, pero no acredita causalidad ni cumplimiento documental.",
  ],
}
