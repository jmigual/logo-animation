from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from manim import (
    Create,
    Dot,
    FadeTransform,
    LaggedStart,
    ManimColor,
    VMobject,
    Scene,
    SVGMobject,
    ValueTracker,
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
    dot_min_spacing = 0.1
    dot_radius_ratio = 0.4
    dot_wave_cycles = 3
    dot_wave_run_time = 3.6
    dot_wave_x_ratio = 0.04
    dot_wave_y_ratio = 0.12

    def construct(self):
        print("Starting")
        drawable_segments = self._load_segments()

        vgroup = VGroup(*[spec.mobject for spec in drawable_segments])
        mobjects = []
        for segment_spec in drawable_segments:
            mobject = segment_spec.mobject
            mobject.set_stroke(color=self.construction_color, width=self.construction_stroke_width)
            mobject.set_fill(opacity=0)
            mobjects.append(Create(mobject, run_time=segment_spec.time, rate_func=rate_functions.ease_in_out_sine))

        self.play(
            LaggedStart(*mobjects, lag_ratio=0.4),
            rate_func=rate_functions.ease_in_out_cubic,
            run_time=5,
        )

        filled_logo = self._build_filled_logo()
        self.add(filled_logo)
        self.play(
            filled_logo.animate.set_fill(opacity=self.fill_opacity),
            vgroup.animate.set_stroke(opacity=0),
            run_time=1,
            rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait(0.6)
        dot_field, home_positions, logo_bounds, dot_spacing = self._build_logo_dot_field()
        self.play(
            FadeTransform(filled_logo, dot_field),
            run_time=2,
            rate_func=rate_functions.ease_in_out_sine,
        )

        self._play_logo_dot_wave(dot_field, home_positions, logo_bounds, dot_spacing)
        self.wait(1)

    def _load_paths(self) -> list[SvgPath]:
        svg = SVG.parse(str(self.asset_path))
        paths: list[SvgPath] = [element for element in svg.elements() if isinstance(element, SvgPath)]
        if not paths:
            raise ValueError(f"No SVG paths found in {self.asset_path}")

        self._configure_projection(paths)
        return paths

    def _load_segments(self) -> list[SegmentSpec]:
        paths = self._load_paths()
        segments = []
        for path in paths:
            for segment in path.segments():
                segment_spec = self._build_segment_spec(segment)

                if isinstance(segment_spec, list):
                    segments.extend(segment_spec)
                elif segment_spec is not None:
                    segments.append(segment_spec)

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

    def _build_segment_spec(self, segment: PathSegment) -> SegmentSpec | None | list[SegmentSpec]:
        if isinstance(segment, Close):
            return self._build_segment_spec_close(segment)
        elif isinstance(segment, Move):
            return None
        elif isinstance(segment, Line):
            return self._build_segment_spec_line(segment)
        elif isinstance(segment, SvgCubicBezier):
            return self._build_segment_spec_cubic_bezier(segment)
        else:
            print(f"Unsupported segment type: {type(segment)}")
            return None

    def _build_segment_spec_close(self, segment: Close) -> SegmentSpec | None:
        if segment.start is None or segment.end is None:
            return None

        # Same as line
        return self._build_segment_spec_line(cast(Line, segment))

    def _build_segment_spec_line(self, segment: Line) -> SegmentSpec | None:
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

    def _build_segment_spec_cubic_bezier(self, segment: SvgCubicBezier) -> SegmentSpec | list[SegmentSpec]:
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

    def _build_logo_dot_field(self) -> tuple[VGroup, np.ndarray, np.ndarray, float]:
        polylines = self._build_logo_polylines(self._load_paths())
        all_points = np.array([point for polyline in polylines for point in polyline], dtype=float)
        min_x, min_y = np.min(all_points[:, :2], axis=0)
        max_x, max_y = np.max(all_points[:, :2], axis=0)

        dot_spacing = self.dot_min_spacing
        edges = [
            (start_point, end_point)
            for polyline in polylines
            for start_point, end_point in zip(polyline, polyline[1:])
            if not np.allclose(start_point, end_point)
        ]

        dot_radius = dot_spacing * self.dot_radius_ratio
        dots = VGroup()
        home_positions = []
        x_values = np.arange(min_x, max_x + dot_spacing, dot_spacing)
        y_values = np.arange(min_y, max_y + dot_spacing, dot_spacing)

        for row_index, y in enumerate(y_values):
            row_offset = dot_spacing * 0.5 if row_index % 2 else 0.0
            for x in x_values + row_offset:
                point = np.array([x, y, 0.0], dtype=float)
                if not self._point_is_inside_logo(point, dot_radius, edges):
                    continue

                dot = Dot(point=point, radius=dot_radius, color=self.fill_color)
                dot.set_stroke(width=0, opacity=0)
                dot.set_z_index(1)
                dots.add(dot)
                home_positions.append(point)

        if not home_positions:
            raise ValueError("The rasterized logo did not produce any dots")

        logo_bounds = np.array([[min_x, min_y], [max_x, max_y]], dtype=float)
        return dots, np.array(home_positions, dtype=float), logo_bounds, dot_spacing

    def _build_logo_polylines(self, paths: list[SvgPath]) -> list[list[np.ndarray]]:
        polylines = []
        for path in paths:
            current_polyline = []
            for segment in path.segments():
                if isinstance(segment, Move):
                    if len(current_polyline) > 1:
                        polylines.append(self._close_polyline(current_polyline))
                    current_polyline = [self._scene_point(segment.end)] if segment.end is not None else []
                    continue

                segment_points = self._approximate_segment_points(segment)
                if not segment_points:
                    continue

                if not current_polyline:
                    current_polyline = list(segment_points)
                elif np.allclose(current_polyline[-1], segment_points[0]):
                    current_polyline.extend(segment_points[1:])
                else:
                    current_polyline.extend(segment_points)

                if isinstance(segment, Close) and len(current_polyline) > 1:
                    polylines.append(self._close_polyline(current_polyline))
                    current_polyline = []

            if len(current_polyline) > 1:
                polylines.append(self._close_polyline(current_polyline))

        if not polylines:
            raise ValueError("The SVG did not produce any closed polylines for rasterization")

        return polylines

    def _close_polyline(self, polyline: list[np.ndarray]) -> list[np.ndarray]:
        closed_polyline = list(polyline)
        if not np.allclose(closed_polyline[0], closed_polyline[-1]):
            closed_polyline.append(closed_polyline[0])
        return closed_polyline

    def _approximate_segment_points(self, segment: PathSegment) -> list[np.ndarray]:
        if isinstance(segment, (Line, Close)):
            if segment.start is None or segment.end is None:
                return []
            return [self._scene_point(segment.start), self._scene_point(segment.end)]

        if isinstance(segment, SvgCubicBezier):
            if segment.start is None or segment.end is None:
                return []

            start_point = self._scene_point(segment.start)
            control1 = self._scene_point(segment.control1)
            control2 = self._scene_point(segment.control2)
            end_point = self._scene_point(segment.end)
            curve_length = max(segment.length() * self.svg_scale, self.dot_min_spacing)
            sample_count = max(6, int(np.ceil(curve_length / (self.dot_min_spacing * 0.75))))

            return [
                ((1 - t) ** 3) * start_point
                + 3 * ((1 - t) ** 2) * t * control1
                + 3 * (1 - t) * (t**2) * control2
                + (t**3) * end_point
                for t in np.linspace(0.0, 1.0, sample_count + 1)
            ]

        return []

    def _point_is_inside_logo(
        self,
        point: np.ndarray,
        radius: float,
        edges: list[tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        intersections = 0
        x, y = point[:2]
        min_edge_distance = float("inf")

        for start_point, end_point in edges:
            x1, y1 = start_point[:2]
            x2, y2 = end_point[:2]
            min_edge_distance = min(min_edge_distance, self._distance_point_to_segment(point, start_point, end_point))

            if np.isclose(y1, y2):
                continue
            if y < min(y1, y2) or y >= max(y1, y2):
                continue

            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if intersection_x >= x:
                intersections += 1

        return intersections % 2 == 1 and min_edge_distance >= radius

    def _distance_point_to_segment(
        self,
        point: np.ndarray,
        start_point: np.ndarray,
        end_point: np.ndarray,
    ) -> float:
        segment = end_point - start_point
        segment_length_squared = float(np.dot(segment[:2], segment[:2]))
        if np.isclose(segment_length_squared, 0.0):
            return float(np.linalg.norm(point[:2] - start_point[:2]))

        projection = float(np.dot(point[:2] - start_point[:2], segment[:2]) / segment_length_squared)
        clamped_projection = np.clip(projection, 0.0, 1.0)
        closest_point = start_point[:2] + clamped_projection * segment[:2]
        return float(np.linalg.norm(point[:2] - closest_point))

    def _play_logo_dot_wave(
        self,
        dot_field: VGroup,
        home_positions: np.ndarray,
        logo_bounds: np.ndarray,
        dot_spacing: float,
    ) -> None:
        width = max(float(logo_bounds[1][0] - logo_bounds[0][0]), 1e-6)
        height = max(float(logo_bounds[1][1] - logo_bounds[0][1]), 1e-6)

        primary_phase = ((home_positions[:, 0] - logo_bounds[0][0]) / width) * (2.5 * np.pi)
        primary_phase += ((home_positions[:, 1] - logo_bounds[0][1]) / height) * np.pi
        secondary_phase = ((home_positions[:, 1] - logo_bounds[0][1]) / height) * (2 * np.pi)
        progress = ValueTracker(0.0)

        def update_dots(group):
            alpha = progress.get_value()
            envelope = np.sin(np.pi * alpha)
            theta = 2 * np.pi * self.dot_wave_cycles * alpha
            x_amplitude = dot_spacing * self.dot_wave_x_ratio
            y_amplitude = dot_spacing * self.dot_wave_y_ratio

            for dot, home_point, phase, secondary in zip(
                group.submobjects,
                home_positions,
                primary_phase,
                secondary_phase,
            ):
                x_offset = envelope * x_amplitude * np.sin((2 * theta) - secondary)
                y_offset = envelope * y_amplitude * np.sin(theta + phase)
                dot.move_to(home_point + np.array([x_offset, y_offset, 0.0]))

        dot_field.add_updater(update_dots)
        self.play(
            progress.animate.set_value(1.0),
            run_time=self.dot_wave_run_time,
            rate_func=rate_functions.linear,
        )
        dot_field.remove_updater(update_dots)
        update_dots(dot_field)
