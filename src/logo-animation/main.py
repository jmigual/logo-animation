from dataclasses import dataclass
from pathlib import Path

import numpy as np
from manim import Create, ManimColor, ParametricFunction, Scene, config, rate_functions
from svgelements import Close, Move, Path as SvgPath, SVG


config.background_color = ManimColor("#f7f2e7")


@dataclass(frozen=True)
class SegmentSpec:
    segment: object
    scaled_length: float
    step_size: float
    start_direction: np.ndarray
    end_direction: np.ndarray


class LogoAnimation(Scene):
    asset_path = Path(__file__).resolve().parents[2] / "assets" / "logo_capital.svg"
    final_color = "#101010"
    construction_color = "#277da1"
    highlight_color = "#f4a261"
    final_stroke_width = 5
    construction_stroke_width = 7
    highlight_stroke_width = 12
    fit_width_ratio = 0.74
    fit_height_ratio = 0.72

    def construct(self):
        drawable_segments = self._load_segments()

        for segment_spec in drawable_segments:
            stretched_segment = self._build_segment_mobject(segment_spec, stretched=True)
            highlighted_segment = self._build_segment_mobject(
                segment_spec,
                color=self.highlight_color,
                stroke_width=self.highlight_stroke_width,
            )

            self.play(
                self._create_animation(stretched_segment, segment_spec.scaled_length),
                rate_func=rate_functions.ease_out_sine,
            )
            self.play(
                stretched_segment.animate.become(highlighted_segment),
                run_time=0.18,
                rate_func=rate_functions.smooth,
            )
            self.play(
                stretched_segment.animate.set_stroke(
                    color=self.final_color,
                    width=self.final_stroke_width,
                ),
                run_time=0.16,
                rate_func=rate_functions.ease_in_out_sine,
            )

        self.wait(0.5)

    def _load_segments(self):
        svg = SVG.parse(str(self.asset_path))
        paths = [element for element in svg.elements() if isinstance(element, SvgPath)]
        if not paths:
            raise ValueError(f"No SVG paths found in {self.asset_path}")

        self._configure_projection(paths)
        segments = []
        for path in paths:
            for segment in path:
                if isinstance(segment, Move):
                    continue
                segments.append(self._build_segment_spec(segment))
        return segments

    def _configure_projection(self, paths):
        bounding_boxes = [path.bbox() for path in paths]
        min_x = min(float(box[0]) for box in bounding_boxes)
        min_y = min(float(box[1]) for box in bounding_boxes)
        max_x = max(float(box[2]) for box in bounding_boxes)
        max_y = max(float(box[3]) for box in bounding_boxes)

        self.svg_center = np.array(
            [
                (min_x + max_x) / 2,
                (min_y + max_y) / 2,
            ],
            dtype=float,
        )
        svg_width = max_x - min_x
        svg_height = max_y - min_y
        self.svg_scale = min(
            config.frame_width * self.fit_width_ratio / svg_width,
            config.frame_height * self.fit_height_ratio / svg_height,
        )
        self.stretch_distance = float(np.hypot(config.frame_width, config.frame_height) * 0.95)

    def _build_segment_spec(self, segment):
        scaled_length = float(segment.length()) * self.svg_scale
        step_size = self._segment_step(segment, scaled_length)
        start_direction = self._segment_tangent(segment, 0.0, 0.02)
        end_direction = self._segment_tangent(segment, 1.0, -0.02)
        return SegmentSpec(
            segment=segment,
            scaled_length=scaled_length,
            step_size=step_size,
            start_direction=start_direction,
            end_direction=end_direction,
        )

    def _segment_step(self, segment, scaled_length):
        if isinstance(segment, Close):
            return 0.08
        if type(segment).__name__ == "Line":
            return 0.08 if scaled_length < 1.4 else 0.05
        if type(segment).__name__ == "Arc":
            return 0.035
        return 0.03

    def _segment_tangent(self, segment, anchor, delta):
        anchor_point = self._scene_point(segment.point(anchor))
        comparison_point = self._scene_point(segment.point(np.clip(anchor + delta, 0.0, 1.0)))
        tangent = comparison_point - anchor_point
        norm = np.linalg.norm(tangent)
        if norm == 0:
            return np.array([1.0, 0.0, 0.0])
        return tangent / norm

    def _scene_point(self, svg_point):
        x = (float(svg_point.x) - self.svg_center[0]) * self.svg_scale
        y = -(float(svg_point.y) - self.svg_center[1]) * self.svg_scale
        return np.array([x, y, 0.0])

    def _stretched_scene_point(self, segment_spec, alpha):
        alpha = float(alpha)
        base_point = self._scene_point(segment_spec.segment.point(alpha))
        start_pull = -segment_spec.start_direction * self.stretch_distance * (1 - alpha) ** 1.35
        end_pull = segment_spec.end_direction * self.stretch_distance * alpha**1.35
        return base_point + start_pull + end_pull

    def _build_segment_mobject(
        self,
        segment_spec,
        stretched=False,
        color=None,
        stroke_width=None,
    ):
        def point_builder(alpha):
            if stretched:
                return self._stretched_scene_point(segment_spec, alpha)
            return self._scene_point(segment_spec.segment.point(alpha))

        curve = ParametricFunction(
            point_builder,
            t_range=(0, 1, segment_spec.step_size),
            use_smoothing=not isinstance(segment_spec.segment, Close),
        )
        curve.set_stroke(
            color=color or (self.construction_color if stretched else self.final_color),
            width=stroke_width or (
                self.construction_stroke_width if stretched else self.final_stroke_width
            ),
        )
        return curve

    def _create_animation(self, stretched_segment, scaled_length):
        run_time = float(np.clip(0.12 + scaled_length * 0.08, 0.18, 0.5))
        return Create(stretched_segment, run_time=run_time)
