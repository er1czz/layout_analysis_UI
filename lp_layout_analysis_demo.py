import layoutparser as lp
import cv2
import numpy as np
import gradio as gr
import time

def lp_fn(model_label, ROI_thresh, image_path):

    start_time = time.perf_counter()
    
    ROI_thresh = float(ROI_thresh)
    # Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)

    #load image
    image = cv2.imread(image_path)
    image = image[..., ::-1]

    #load model
    model_map = {"Layout: faster_rcnn": "PubLayNet: faster_rcnn_R_50_FPN_3x", 
                 "Layout: mask_rcnn1": "PubLayNet: mask_rcnn_R_50_FPN_3x", 
                 "Layout: mask_rcnn2": "PubLayNet: mask_rcnn_X_101_32x8d_FPN_3x",
                 "Layout: mask_rcnn3": "PrimaLayout: mask_rcnn_R_50_FPN_3x",
                 "Table: faster-rcnn1": "TableBank: faster_rcnn_R_50_FPN_3x", 
                 "Table: faster-rcnn2": "TableBank: faster_rcnn_R_101_FPN_3x"}

    model_name = model_map[model_label]

    if model_label == "Layout: faster_rcnn":
        model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
                                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", ROI_thresh],
                                         label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

    if model_label == "Layout: mask_rcnn1":
        model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config', 
                                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", ROI_thresh],
                                         label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

    if model_label == "Layout: mask_rcnn2":
        model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config', 
                                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", ROI_thresh],
                                         label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        
    if model_label == "Layout: mask_rcnn3":
        model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config', 
                                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", ROI_thresh],
                                         label_map={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"})
        
    if model_label == "Table: faster-rcnn1":
        model = lp.Detectron2LayoutModel('lp://TableBank/faster_rcnn_R_50_FPN_3x/config',
                                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", ROI_thresh],
                                         label_map={0: "Table"})

    if model_label == "Table: faster-rcnn2":
        model = lp.Detectron2LayoutModel('lp://TableBank/faster_rcnn_R_101_FPN_3x/config',
                                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", ROI_thresh],
                                         label_map={0: "Table"})
    
    layout = model.detect(image)

    output_img = lp.draw_box(image, layout, box_width=3, box_alpha=0.3, show_element_type=True, 
                id_font_size=30, id_text_background_color='white',id_text_background_alpha=0.7,
                color_map = {"Text":"red", "Title":"blue", "List":"green", "Table":"purple", "Figure":"pink"})

    # seconds
    run_time = time.perf_counter() - start_time
    
    return model_name, run_time, output_img


title = "Interactive demo: Document Layout Analysis with LayoutParser (Detectron2)"
description = "Demo for LayoutParser: a unified toolkit for Deep Learning Based Document Image Analysis. To use it, simply upload an image or use the example image below and click 'Submit'. Results will show up in a few seconds. If you want to make the output bigger, right-click on it and select 'Open image in new tab'."
article="<h5>LayoutParser Model Zoo:</h5><n>[https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html](https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html),</n>\
<h5>_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05</h5>\
<n>Only used on test mode</n>\
<n>Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to balance obtaining high recall with not having too many low precision detections that will slow down inference post processing steps (like NMS). \
A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down inference.</s> \
 <h5>https://detectron2.readthedocs.io/en/latest/modules/config.html</h5>"
#css = ".output-image, .input-image, .image-preview {height: 600px !important}"

iface = gr.Interface(fn=lp_fn,
                     inputs=[gr.Dropdown(
                                ["Layout: faster_rcnn", "Layout: mask_rcnn1", "Layout: mask_rcnn2", "Layout: mask_rcnn3", "Table: faster-rcnn1", "Table: faster-rcnn2"], 
                                label="Select use case and model type"),
			     gr.Text(placeholder="0.8", label='from 0 to 1 threshold value for IoU'), 
                             gr.Image(type='filepath', label='Doc image input')],
                     outputs=[gr.Text(label='model name'),
                              gr.Text(label='run time (seconds)'),
                              gr.Image(label="annotated doc image")],
                     title=title,
                     description=description,
                     #examples=examples,
                     article=article,
                     #css=css,
                     allow_flagging="never",
                     batch=True, max_batch_size=50).queue(concurrency_count=4)

if __name__ == '__main__':
    iface.launch(share=False)
