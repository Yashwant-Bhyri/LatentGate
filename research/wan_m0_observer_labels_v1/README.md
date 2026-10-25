# Wan M0 observer-label packet

`blinded_annotation_package.zip` contains two anonymized copies of the same 17-frame saved Wan2.1 sample. Give one copy to each independent rater. Do not share `PRIVATE_UNBLIND_KEY.json` with raters; it only maps the anonymous copy IDs to the source arm.

Each rater should fill the blank columns in `annotation_template.csv` after inspecting every frame. Mark whether the target is visible, visible instance count, whether it is the same object as frame 0, and a tight pixel-edge bounding box (`x1,y1,x2,y2`; right and bottom edges are exclusive). Leave box coordinates blank if the target is absent. Add notes for occlusion, blur, ambiguity, or duplicates. The scored center is the midpoint of the tight box.

Keep the raters blind to SAM2 masks and red-threshold tracks until their labels are returned. This convenience sample can compare the observer proposals with human boxes on one generated clip. It is not the planned multi-seed observer gate and says nothing about policy efficacy.
