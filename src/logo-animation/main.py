from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from manim import (
    Create,
    LaggedStart,
    ManimColor,
    VMobject,
    Scene,
    SVGMobject,
    VGroup,
    config,
    rate_functions,
)
from svgelements import (
    Close,
    Line,
    Move,
    CubicBezier as SvgCubicBezier,
    Path as SvgPath,
    SVG,
    PathSegment,
)


config.background_color = ManimColor("#f7f2e7")


@dataclass(frozen=True)
class SegmentSpec:
    segment: object
    mobject: VMobject
    time: float


class LogoAnimation(Scene):
    asset_path = Path(__file__).resolve().parents[2] / "assets" / "logo_capital.svg"
    construction_color = "#5D5D5D"
    fill_color = "#101010"
    fill_opacity = 1.0
    final_stroke_width = 5
    construction_stroke_width = 2
    highlight_stroke_width = 12
    fit_width_ratio = 0.74
    fit_height_ratio = 0.72
    length_time_ratio = 0.1

    def construct(self):
        drawable_segments = self._load_segments()

        vgroup = VGroup(*[spec.mobject for spec in drawable_segments])
        mobjects = []
        for segment_spec in drawable_segments:
            mobject = segment_spec.mobject
            mobject.set_stroke(color=self.construction_color, width=self.construction_stroke_width)
            mobject.set_fill(opacity=0)
            mobjects.append(Create(mobject, run_time=segment_spec.time))

        self.play(
            LaggedStart(*mobjects, lag_ratio=0.3),
            run_time=5,
            rate_func=rate_functions.ease_in_out_sine,
        )

        filled_logo = self._build_filled_logo()
        self.add(filled_logo)
        self.play(
            filled_logo.animate.set_fill(opacity=self.fill_opacity),
            vgroup.animate.set_stroke(opacity=0),
            run_time=0.55,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait(2)

    def _load_segments(self) -> list[SegmentSpec]:
        svg = SVG.parse(str(self.asset_path))
        paths: list[SvgPath] = [element for element in svg.elements() if isinstance(element, SvgPath)]
        if not paths:
            raise ValueError(f"No SVG paths found in {self.asset_path}")

        self._configure_projection(paths)
        segments = []
        for path in paths:
            for segment in path.segments():
                segment_spec = self._build_segment_spec(segment)
                segments.append(segment_spec)

        return list(filter(lambda x: x is not None, segments))

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

    def _build_segment_spec(self, segment: PathSegment):
        if isinstance(segment, Close):
            return self._build_segment_spec_close(segment)
        elif isinstance(segment, Move):
            return None
        elif isinstance(segment, Line):
            return self._build_segment_spec_line(segment)
        elif isinstance(segment, SvgCubicBezier):
            return self._build_segment_spec_cubic_bezier(segment)
        else:
            return None

    def _build_segment_spec_close(self, segment: Close):
        if segment.start is None or segment.end is None:
            return None

        # Same as line
        return self._build_segment_spec_line(cast(Line, segment))

    def _build_segment_spec_line(self, segment: Line):
        start_point = self._scene_point(segment.start)
        end_point = self._scene_point(segment.end)
        direction = end_point - start_point
        length = float(np.linalg.norm(direction))
        if np.isclose(length, 0.0):
            return None

        # Extend the line until it intersects both sides of the frame.
        frame_x_radius = config.frame_width / 2
        frame_y_radius = config.frame_height / 2
        tolerance = 1e-6
        intersections = []

        if not np.isclose(direction[0], 0.0):
            for x in (-frame_x_radius, frame_x_radius):
                scale = (x - start_point[0]) / direction[0]
                candidate = start_point + direction * scale
                if -frame_y_radius - tolerance <= candidate[1] <= frame_y_radius + tolerance:
                    intersections.append(candidate)

        if not np.isclose(direction[1], 0.0):
            for y in (-frame_y_radius, frame_y_radius):
                scale = (y - start_point[1]) / direction[1]
                candidate = start_point + direction * scale
                if -frame_x_radius - tolerance <= candidate[0] <= frame_x_radius + tolerance:
                    intersections.append(candidate)

        unique_intersections = []
        for candidate in intersections:
            if not any(np.allclose(candidate, existing) for existing in unique_intersections):
                unique_intersections.append(candidate)

        chosen_start, chosen_end = start_point, end_point
        line_mobject = VMobject()
        if len(unique_intersections) >= 2:
            unique_intersections.sort(key=lambda candidate: np.dot(candidate[:2] - start_point[:2], direction[:2]))
            chosen_start, chosen_end = unique_intersections[0], unique_intersections[-1]
        else:
            chosen_start, chosen_end = start_point, end_point

        line_mobject.set_points_as_corners([chosen_start, chosen_end])

        length = float(np.linalg.norm(chosen_end - chosen_start))
        return SegmentSpec(segment=segment, mobject=line_mobject, time=length * self.length_time_ratio)

    def _build_segment_spec_cubic_bezier(self, segment: SvgCubicBezier):
        start_point = self._scene_point(segment.start)
        control1 = self._scene_point(segment.control1)
        control2 = self._scene_point(segment.control2)
        end_point = self._scene_point(segment.end)
        length = segment.length() * self.svg_scale

        mobject = VMobject()
        mobject.add_cubic_bezier_curve(start_point, control1, control2, end_point)
        return SegmentSpec(
            segment=segment,
            mobject=mobject,
            time=max(length * self.length_time_ratio, 0.04),
        )

    def _scene_point(self, svg_point):
        x = (float(svg_point.x) - self.svg_center[0]) * self.svg_scale
        y = -(float(svg_point.y) - self.svg_center[1]) * self.svg_scale
        return np.array([x, y, 0.0])

    def _build_filled_logo(self):
        filled_logo = SVGMobject(str(self.asset_path))
        filled_logo.set_stroke(width=0, opacity=0)
        filled_logo.set_fill(color=self.fill_color, opacity=0)
        filled_logo.scale_to_fit_height(config.frame_height * self.fit_height_ratio)
        filled_logo.move_to(np.array([0, 0, 0]))
        filled_logo.set_z_index(0)
        return filled_logo
