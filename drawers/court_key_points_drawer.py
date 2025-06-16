import supervision as sv

class CourtKeypointDrawer:
    def __init__(self):
        self.keypoint_color = '#ff2c2c'
        self.edge_color = '#ff9900'
        self.edge_thickness = 2

    def draw(self, frames, court_keypoints):
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=20
        )

        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex(self.edge_color),
            thickness=self.edge_thickness
        )

        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            keypoints = court_keypoints[index]

            # ➤ 加入 edge 畫線（先畫線，再畫點）
            annotated_frame = edge_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints
            )

            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints
            )

            keypoints_numpy = keypoints.cpu().numpy()
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_numpy
            )

            output_frames.append(annotated_frame)

        return output_frames