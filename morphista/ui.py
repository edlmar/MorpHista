"""The desktop window (NiceGUI in native mode)."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

from nicegui import app, run, ui

from . import viewer
from .config import APP_DIR, IMAGE_EXTS, MODELS_DIR, Config, load_settings, save_settings
from .models import DEVICE_LABEL, get_model, list_models
from .pipelines import CSV_FIELDS, Result, collect_images, run_cell_seg, run_detection, run_tissue

STATIC = Path(__file__).parent / "static"
ICON = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "assets" / "icon.ico"  # bundled data lives in _MEIPASS
OSD_URL = "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"
FONT_URL = "https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&display=swap"

CELL_SHORT = {"icc": "ICC", "astrocytes": "Astrocytes"}
COLUMNS = [
    ("Image", "Image"),
    ("Cells", "Cells"),
    ("Density_cells_per_mm2", "Cells / mm²"),
    ("Tissue_Area_mm2", "Tissue mm²"),
    ("Total_Area_mm2", "Image mm²"),
    ("White_Matter", "White"),
    ("Grey_Matter", "Grey"),
    ("Classes", "Classes"),
    ("Pixel_Size_um", "µm / px"),
    ("Time_s", "Sec"),
]
THEME_ICONS = {None: "brightness_auto", True: "dark_mode", False: "light_mode"}
FIELD = "outlined dense hide-bottom-space"
# the banner / app icon mark: a detection box around a cell
LOGO = (  # stroke edges on whole pixels (0 and 3) so it stays sharp at 100 % and 200 % scaling
    '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true" shape-rendering="geometricPrecision">'
    '<rect x="1.5" y="1.5" width="17" height="17" rx="4.5" fill="none" stroke="currentColor" stroke-width="3"/>'
    '<circle cx="10" cy="10" r="3.75" fill="currentColor"/></svg>'
)
ICON_BTN = "flat round dense color=mhmuted"
PRIMARY_BTN = "unelevated no-caps color=mhacc text-color=onacc"
GHOST_BTN = "flat no-caps color=mhink"
ZOOMBAR = (
    '<div class="mh-zoombar">'
    '<button data-act="out" title="Zoom out" aria-label="Zoom out"><i class="material-icons">remove</i></button>'
    '<span class="lvl" title="Scroll to zoom, drag to move, double-click to zoom in">—</span>'
    '<button data-act="in" title="Zoom in" aria-label="Zoom in"><i class="material-icons">add</i></button>'
    '<button data-act="one" class="one" title="Actual pixels" aria-label="Actual pixels">1:1</button>'
    '<button data-act="fit" title="Fit to window" aria-label="Fit to window">'
    '<i class="material-icons">fit_screen</i></button></div>'
)
EMPTY = Result(
    title="Results appear here after a run",
    big="—",
    unit="cells detected",
    summary="",
    image=None,
    rows=[("per mm² tissue", "—"), ("tissue, mm²", "—"), ("µm per pixel", "—")],
)
HELP = (
    "Put `.pth` (RF-DETR) or `.onnx` models anywhere under `models/`. Folder or file names "
    "containing *icc* or *astro* tie a model to a cell type; *seg* marks segmentation models.\n\n"
    "**Detect and count** finds cells, measures the tissue area and reports cells per mm² of tissue. "
    "**Tissue mask** shows what counts as tissue. **Segment cells** saves one mask per cell.\n\n"
    "Results are saved next to the images, in `Detections/`, `Masks/` and `CellInstanceMasks/`. "
    "Pixel size comes from TIFF tags unless you force your own."
)


class Job:
    """State of the running analysis, shared by every open page."""

    def __init__(self):
        self.running = False
        self.cancel = False
        self.progress = 0.0
        self.status = "Ready"
        self.lines: list[str] = []  # each page remembers how far it has read

    def log(self, msg: str) -> None:
        self.lines.append(f"{time.strftime('%H:%M:%S')}  {msg}")


settings = load_settings()
if "dark" in settings.get("config", {}):  # older versions kept the theme inside the analysis settings
    settings.setdefault("dark", settings["config"].pop("dark"))
cfg = Config.from_dict(settings.get("config", {}))
job = Job()


def persist() -> None:
    settings["config"] = asdict(cfg)
    save_settings(settings)


def to_int(v) -> int:
    return int(v or 0)


def to_float(v) -> float:
    return float(v or 0)


def model_options(attr: str) -> dict[str, str]:
    """Models for the current cell type; points cfg.<attr> at the first one if its choice isn't listed."""
    found = list_models(cfg.cell_type, segmentation=attr == "seg_model_path")
    opts = {str(p): " / ".join(p.relative_to(MODELS_DIR).parts) for p in found}
    if getattr(cfg, attr) not in opts:
        setattr(cfg, attr, next(iter(opts), None))
    return opts


async def file_dialog(kind: str, **kwargs) -> str | None:
    """Native OPEN / FOLDER / SAVE dialog; one path or None."""
    import webview

    enum = getattr(webview, "FileDialog", None)  # pywebview >= 6; older: *_DIALOG constants
    dialog_type = getattr(enum, kind) if enum else getattr(webview, f"{kind}_DIALOG")
    result = await app.native.main_window.create_file_dialog(dialog_type, **kwargs)
    if not result:
        return None
    return result if isinstance(result, str) else result[0]


class MainPage:
    def __init__(self):
        self.shown: Result = EMPTY
        self.shown_file = "No image yet"
        self.seen_log = 0
        ui.add_head_html(
            f'<link rel="preconnect" href="https://fonts.googleapis.com">'
            f'<link href="{FONT_URL}" rel="stylesheet"><script src="{OSD_URL}"></script>'
        )
        ui.add_css((STATIC / "style.css").read_text(encoding="utf-8"))
        ui.add_body_html(f"<script>{(STATIC / 'viewer.js').read_text(encoding='utf-8')}</script>")
        ui.colors(primary="#047857", negative="#B42318")
        self.dark = ui.dark_mode(settings.get("dark"))

        self._dialogs()
        with ui.element("div").classes("mh-app"):
            self._top_bar()
            self._sidebar()
            with ui.element("div").classes("mh-main"):
                with ui.element("div").classes("mh-work"):
                    self._viewer()
                    self.metrics()
                self._results()

        ui.timer(0.3, self.drain_log)
        self.cell_toggle.on_value_change(self.refresh_models)
        self.refresh_models()

    # ---------------------------------------------------------------- layout
    def _dialogs(self):
        with ui.dialog() as self.help_dialog, ui.card().classes("max-w-md p-6 gap-3"):
            ui.label("How MorpHista works").classes("text-base font-semibold")
            ui.markdown(HELP).classes("text-sm")

        with ui.dialog() as self.profile_dialog, ui.card().classes("w-96 p-6 gap-4"):
            ui.label("Settings profiles").classes("text-base font-semibold")
            self.profile_select = ui.select([], label="Saved profiles").props(FIELD).classes("w-full")
            with ui.row().classes("w-full gap-2"):
                ui.button("Load", on_click=self.load_profile).props(PRIMARY_BTN)
                ui.button("Delete", on_click=self.delete_profile).props("flat no-caps color=mhdanger")
            ui.separator()
            with ui.row().classes("w-full no-wrap gap-2 items-center"):
                self.profile_name = ui.input("Save current settings as").props(FIELD).classes("grow")
                ui.button("Save", on_click=self.save_profile).props(PRIMARY_BTN)

    def _top_bar(self):
        with ui.element("div").classes("mh-top"):
            with ui.element("div").classes("mh-brand"):
                ui.html(LOGO).classes("text-mhacc flex")
                ui.label("MorpHista")
            ui.space()
            ui.label(DEVICE_LABEL).classes("mh-device")
            self.theme_btn = (
                ui.button(icon=THEME_ICONS[settings.get("dark")], on_click=self.cycle_theme)
                .props(ICON_BTN)
                .tooltip("Theme: system, dark, light")
            )
            ui.button(icon="bookmark_border", on_click=self.open_profiles).props(ICON_BTN).tooltip("Settings profiles")
            ui.button(icon="help_outline", on_click=self.help_dialog.open).props(ICON_BTN).tooltip("How it works")

    def _sidebar(self):
        with ui.element("div").classes("mh-side"):
            with ui.element("div").classes("mh-side-body"):
                with ui.element("div").classes("mh-group"):
                    ui.label("Cell type").classes("mh-caption")
                    self.cell_toggle = (
                        ui.toggle(CELL_SHORT)
                        .bind_value(cfg, "cell_type")
                        .props(
                            "spread no-caps unelevated toggle-color=mhseg toggle-text-color=mhink text-color=mhmuted"
                        )
                        .classes("w-full mh-seg")
                    )

                with ui.element("div").classes("mh-group"):
                    ui.label("Models").classes("mh-caption")
                    self.det_select = (
                        ui.select(model_options("model_path"), label="Detection")
                        .bind_value(cfg, "model_path")
                        .props(FIELD + " options-dense")
                        .classes("w-full")
                    )
                    self.seg_select = (
                        ui.select(model_options("seg_model_path"), label="Segmentation")
                        .bind_value(cfg, "seg_model_path")
                        .props(FIELD + " options-dense")
                        .classes("w-full")
                    )

                with ui.element("div").classes("mh-group"):
                    ui.label("Images").classes("mh-caption")
                    with (
                        ui.input(placeholder="File or folder")
                        .bind_value(cfg, "input_path")
                        .props(FIELD)
                        .classes("w-full") as path_input,
                        path_input.add_slot("append"),
                    ):
                        ui.button(icon="folder_open", on_click=lambda: self.pick(folder=True)).props(
                            ICON_BTN + " size=sm"
                        ).tooltip("Choose a folder")
                        ui.button(icon="image", on_click=lambda: self.pick(folder=False)).props(
                            ICON_BTN + " size=sm"
                        ).tooltip("Choose an image")

                with ui.element("div").classes("mh-group"):
                    run_btn = (
                        ui.button("Detect and count", on_click=lambda: self.start("detect"))
                        .props(PRIMARY_BTN)
                        .classes("mh-run w-full")
                    )
                    with ui.row().classes("w-full no-wrap gap-2"):
                        tissue_btn = (
                            ui.button("Tissue mask", on_click=lambda: self.start("tissue"))
                            .props(GHOST_BTN)
                            .classes("mh-ghost")
                        )
                        seg_btn = (
                            ui.button("Segment cells", on_click=lambda: self.start("seg"))
                            .props(GHOST_BTN)
                            .classes("mh-ghost")
                        )
                    for b in (run_btn, tissue_btn, seg_btn):
                        b.bind_enabled_from(job, "running", backward=lambda r: not r)

                with ui.element("div").classes("mh-settings"):
                    self._tissue_settings()
                    self._detection_settings()

            with ui.element("div").classes("mh-status"):
                ui.label().bind_text_from(job, "status").classes("txt")
                ui.button("Cancel", on_click=lambda: setattr(job, "cancel", True)).props(
                    "flat dense no-caps color=mhdanger"
                ).bind_visibility_from(job, "running")

    @staticmethod
    def _tissue_settings():
        with ui.expansion("Tissue mask settings").props("dense"), ui.grid(columns=2).classes("w-full gap-2"):
            ui.number("Brightness cut-off", min=0, max=255, precision=0).props(FIELD).bind_value(
                cfg, "threshold", forward=to_int
            ).tooltip("Pixels darker than this count as tissue")
            ui.number("Smallest piece, mm²", min=0, step=0.1).props(FIELD).bind_value(
                cfg, "min_area_mm2", forward=to_float
            )
            ui.number("Pixel size, µm", min=0.001, step=0.01, format="%.3f").props(FIELD).bind_value(
                cfg, "pixel_size_um", forward=to_float
            ).tooltip("Used when the image has no resolution tags")
            ui.checkbox("Ignore TIFF tags").props("dense").bind_value(cfg, "force_pixel_size").classes(
                "self-center text-sm"
            )
            ui.number("Mask opacity", min=0, max=1, step=0.05).props(FIELD).bind_value(
                cfg, "mask_alpha", forward=to_float
            )
            ui.color_input("Mask color").props(FIELD).bind_value(cfg, "mask_color")

    @staticmethod
    def _detection_settings():
        with ui.expansion("Detection settings").props("dense"), ui.grid(columns=2).classes("w-full gap-2"):
            ui.number("Min confidence", min=0.01, max=1, step=0.05).props(FIELD).bind_value(
                cfg, "confidence", forward=to_float
            )
            ui.number("Overlap to merge", min=0.01, max=1, step=0.05).props(FIELD).bind_value(
                cfg, "nms", forward=to_float
            ).tooltip("Boxes overlapping more than this (IoU) become one cell")
            ui.number("Merge within, px", min=0, precision=0).props(FIELD).bind_value(
                cfg, "merge_px", forward=to_int
            ).tooltip("Same-class boxes closer than this become one cell")
            ui.number("Box line, px", min=1, max=10, precision=0).props(FIELD).bind_value(
                cfg, "box_thickness", forward=to_int
            )
            ui.number("Tile size, px", min=0, precision=0).props(FIELD).bind_value(
                cfg, "tile_size", forward=to_int
            ).tooltip("0 uses the model's input size")
            ui.number("Tile overlap, px", min=0, precision=0).props(FIELD).bind_value(
                cfg, "tile_overlap", forward=to_int
            ).tooltip("0 picks 15 % of the tile")
            ui.color_input("Box color").props(FIELD).bind_value(cfg, "box_color")
            ui.checkbox("Skip background tiles").props("dense").bind_value(cfg, "skip_background").classes(
                "self-center text-sm"
            ).tooltip(
                "Only tiles with at least 1 % tissue go to the model. Much faster on slides; "
                "also drops detections on isolated specks in the background."
            )

    def _viewer(self):
        with ui.element("div").classes("mh-viewer"):
            ui.linear_progress(show_value=False, size="2px", color="mhacc").classes("mh-progress").bind_value_from(
                job, "progress"
            ).bind_visibility_from(job, "running")
            with ui.element("div").classes("mh-empty") as self.empty:
                ui.icon("add_photo_alternate", size="40px").classes("text-mhmuted")
                ui.label("Choose what to analyse").classes("t")
                ui.label("A single image or a whole folder. The latest result is shown here.").classes("s")
                with ui.row().classes("gap-2 mt-3"):
                    ui.button("Choose image", on_click=lambda: self.pick(folder=False)).props(PRIMARY_BTN).classes(
                        "rounded-lg"
                    )
                    ui.button("Choose folder", on_click=lambda: self.pick(folder=True)).props(GHOST_BTN).classes(
                        "mh-ghost rounded-lg px-4"
                    )
            self.osd = ui.element("div").props("id=mh-osd").classes("mh-osd hidden")
            self.zoombar = ui.html(ZOOMBAR).classes("hidden")

    @ui.refreshable_method
    def metrics(self):
        r, idle = self.shown, self.shown is EMPTY
        with ui.element("div").classes("mh-metrics" + (" idle" if idle else "")):
            ui.label(self.shown_file).classes("file").tooltip(self.shown_file)
            ui.label(r.title).classes("kind")
            ui.label(r.big).classes("big num")
            ui.label(r.unit).classes("unit")
            for name, value in r.rows:
                with ui.element("div").classes("row"):
                    ui.label(name).classes("k")
                    ui.label(value).classes("v num")

    def _results(self):
        with ui.element("div").classes("mh-results"):
            with ui.element("div").classes("mh-results-head"):
                with ui.tabs().props("dense no-caps align=left indicator-color=mhacc narrow-indicator") as self.tabs:
                    self.t_res = ui.tab("results", label="Results")
                    self.t_log = ui.tab("log", label="Log")
                ui.space()
                ui.button("Export CSV", icon="download", on_click=self.export_csv).props(GHOST_BTN + " dense").classes(
                    "px-2"
                )
                ui.button(icon="delete_sweep", on_click=self.clear_rows).props(ICON_BTN).tooltip("Clear the table")
            with ui.tab_panels(self.tabs, value=self.t_res).props("animated=false"):
                with ui.tab_panel(self.t_res):
                    self.table = (
                        ui.table(
                            columns=[
                                {
                                    "name": f,
                                    "label": h,
                                    "field": f,
                                    "sortable": True,
                                    "align": "left" if f in ("Image", "Classes") else "right",
                                }
                                for f, h in COLUMNS
                            ],
                            rows=[],
                            row_key="Image",
                            pagination=0,
                        )
                        .props("dense flat virtual-scroll hide-pagination")
                        .classes("w-full h-full")
                    )
                    with self.table.add_slot("no-data"):
                        ui.label("Run Detect and count to fill this table, one row per image.").classes("mh-none")
                with ui.tab_panel(self.t_log):
                    self.log = ui.log(max_lines=500).classes("mh-log")

    # ---------------------------------------------------------------- handlers
    def refresh_models(self):
        for select, attr, name in (
            (self.det_select, "model_path", "Detection"),
            (self.seg_select, "seg_model_path", "Segmentation"),
        ):
            opts = model_options(attr)
            select.props(f'label="{name}{"" if opts else " (none in models/)"}"')
            select.set_options(opts, value=getattr(cfg, attr))

    def cycle_theme(self):
        settings["dark"] = {None: True, True: False, False: None}[settings.get("dark")]
        self.dark.value = settings["dark"]
        self.theme_btn.props(f"icon={THEME_ICONS[settings['dark']]}")
        persist()

    def open_profiles(self):
        self.profile_select.set_options(sorted(settings.get("profiles", {})))
        self.profile_dialog.open()

    def save_profile(self):
        name = self.profile_name.value
        if not name:
            return ui.notify("Type a name for the profile", type="warning")
        settings.setdefault("profiles", {})[name] = asdict(cfg)
        persist()
        self.profile_select.set_options(sorted(settings["profiles"]))
        ui.notify(f"Profile “{name}” saved")

    def load_profile(self):
        name = self.profile_select.value
        data = settings.get("profiles", {}).get(name)
        if not data:
            return ui.notify("Choose a profile to load", type="warning")
        for k, v in asdict(Config.from_dict(data)).items():
            setattr(cfg, k, v)
        self.refresh_models()
        self.profile_dialog.close()
        ui.notify(f"Profile “{name}” loaded")

    def delete_profile(self):
        settings.get("profiles", {}).pop(self.profile_select.value, None)
        persist()
        self.profile_select.set_options(sorted(settings.get("profiles", {})))

    def clear_rows(self):
        self.table.rows.clear()
        self.table.update()

    def show(self, path: Path, result: Result):
        self.shown, self.shown_file = result, path.name
        self.metrics.refresh()
        tiles = viewer.publish(result.image)
        self.osd.classes(remove="hidden")
        self.zoombar.classes(remove="hidden")
        self.empty.set_visibility(False)
        ui.run_javascript(f"mhShow('{tiles.key}', {tiles.w}, {tiles.h})")

    async def pick(self, folder: bool):
        exts = ";".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        path = (
            await file_dialog("FOLDER")
            if folder
            else await file_dialog("OPEN", file_types=(f"Images ({exts})", "All files (*.*)"))
        )
        if path:
            cfg.input_path = path

    async def export_csv(self):
        rows = self.table.rows
        if not rows:
            return ui.notify("Nothing to export yet — run Detect and count first", type="warning")
        path = await file_dialog("SAVE", save_filename="morphista_results.csv")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        ui.notify(f"Exported {len(rows)} rows to {Path(path).name}")

    async def start(self, kind: str):
        images = collect_images(cfg.input_path)
        if not images:
            return ui.notify("No images at that path. Choose a file or a folder with images.", type="warning")
        model_path = {"detect": cfg.model_path, "seg": cfg.seg_model_path}.get(kind)
        if kind != "tissue" and not model_path:
            which = "detection" if kind == "detect" else "segmentation"
            return ui.notify(f"No {which} model for {CELL_SHORT[cfg.cell_type]} in models/", type="warning")
        persist()
        job.running, job.cancel, job.progress = True, False, 0.0
        job.lines.clear()
        try:
            model = None
            if kind != "tissue":
                job.status = "Loading model…"
                try:
                    model = await run.io_bound(get_model, model_path, kind == "seg")
                except Exception as e:
                    job.log(f"Model failed to load: {e}")
                    self.tabs.set_value(self.t_log)
                    return ui.notify(f"Model failed to load: {e}", type="negative", multi_line=True)
                job.log(f"Model: {model.kind}")
            if kind == "detect":
                self.clear_rows()
                self.tabs.set_value(self.t_res)
            await self._run_all(kind, images, model)
        finally:
            job.running = False

    async def _run_all(self, kind: str, images: list[Path], model):
        n, errors = len(images), 0
        for i, path in enumerate(images):
            if job.cancel:
                job.log("Cancelled")
                break

            def progress(frac: float, msg: str, i=i, path=path):
                job.progress = (i + frac) / n
                job.status = f"{i + 1} of {n} · {path.name} · {msg}"

            try:
                if kind == "detect":
                    result = await run.io_bound(run_detection, path, model, cfg, progress, lambda: job.cancel)
                    self.table.add_row(result.table_row)
                elif kind == "seg":
                    result = await run.io_bound(run_cell_seg, path, model, cfg, progress, lambda: job.cancel)
                else:
                    result = await run.io_bound(run_tissue, path, cfg, progress, lambda: job.cancel)
                self.show(path, result)
                job.log(f"{path.name}: {result.summary}")
            except Exception as e:
                errors += 1
                job.log(f"{path.name}: failed — {e}")
        job.progress = 1.0
        job.status = f"Finished {n} image{'s' * (n > 1)}" + (f", {errors} failed — see Log" if errors else "")
        if errors:
            self.tabs.set_value(self.t_log)

    def drain_log(self):
        if self.seen_log > len(job.lines):
            self.log.clear()
            self.seen_log = 0
        for line in job.lines[self.seen_log :]:
            self.log.push(line)
        self.seen_log = len(job.lines)


@ui.page("/", title="MorpHista")
def index():
    MainPage()


def main():
    app.on_shutdown(persist)
    ui.run(title="MorpHista", favicon=ICON, native=True, window_size=(1400, 900), reload=False)
