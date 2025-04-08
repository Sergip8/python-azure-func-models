import os
from transformers import logging
import numpy as np
from transformers import AutoProcessor, AutoModelForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
from manga_ocr import MangaOcr
import deepl

class MangaTranslator:
    def __init__(self):
        # Initialize Manga OCR
        print("Initializing Manga OCR (this may take a moment)...")
        self.mocr = MangaOcr()
        self.processor = AutoProcessor.from_pretrained("ogkalu/comic-text-and-bubble-detector", trust_remote_code=True)
        self.model = AutoModelForObjectDetection.from_pretrained("ogkalu/comic-text-and-bubble-detector", trust_remote_code=True)
        logging.set_verbosity_error() 
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.translator = deepl.Translator(os.getenv('DEEPL_API_KEY'))
        
        # Font settings for translated text
        self.font_path = "arial.ttf"  # Default font, change as needed
        self.font_size = 11

    def remove_contained_boxes(self, boxes):
        """
        Eliminates boxes that are completely contained within other boxes.
        Returns the indices of the boxes that should be kept.
        """
        keep = []
        num_boxes = boxes.shape[0]
        
        for i in range(num_boxes):
            contained = False
            box_i = boxes[i]
            
            for j in range(num_boxes):
                if i == j:
                    continue
                    
                box_j = boxes[j]
                # Check if box_i contains box_j (not the other way around)
                if (box_j[0] >= box_i[0] and box_j[1] >= box_i[1] and 
                    box_j[2] <= box_i[2] and box_j[3] <= box_i[3]):
                    contained = True
                    break
                    
            if not contained:
                keep.append(i)
        
        return torch.tensor(keep, dtype=torch.long)
        
    def detect_text_regions(self, image):
        """Detect text regions in a manga image"""
        # Use PIL's optimized loader with reduced size if image is very large
         
        
        # Resize if image is very large (preserves aspect ratio)
        max_dim = 1024
        if max(image.size) > max_dim:
            scale = max_dim / max(image.size)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Convert to RGB
        image = image.convert("RGB")

        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference with mixed precision if available
        if hasattr(torch.cuda, 'amp') and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        # Process results
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (height, width)
        results = self.processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
        
        # Move results back to CPU for further processing
        results = {k: v.cpu() for k, v in results.items()}
        
        # Get indices of boxes to keep
        keep_indices = self.remove_contained_boxes(results['boxes'])

        # Filter all data using these indices
        filtered_result = {
            'boxes': results['boxes'][keep_indices]
        }

        return filtered_result
    
    def convert_to_xywh(self, boxes):
        """Convert [x1, y1, x2, y2] format to (x, y, w, h)"""
        # Extract coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate width and height
        w = x2 - x1
        h = y2 - y1
        
        # Create list of (x, y, w, h) tuples
        xywh_boxes = [(x1[i].item(), y1[i].item(), w[i].item(), h[i].item()) for i in range(len(x1))]
        return xywh_boxes
        
    def merge_overlapping_regions(self, regions, iou_threshold=0.3):
        """Merge regions using Non-Maximum Suppression"""
        if not regions:
            return []

        boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in regions])
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while idxs.size > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], 
                np.where(overlap > iou_threshold)[0])))

        return [regions[i] for i in pick]

    def recognize_and_translate(self, img, regions, target_lang='en'):
        """Extract text from regions, translate it, and overlay on image"""
        
        result_img = img.copy()
        draw = ImageDraw.Draw(result_img)
        
        # Initialize translator
        
        
        translations = []
        
        for i, (x, y, w, h) in enumerate(regions):
            # Crop the region from the original image
            region_img = img.crop((x, y, x+w, y+h))
            
            try:
                # Extract text using Manga OCR
                text = self.mocr(region_img)
                
                if not text.strip().startswith("．") and len(text.strip()) >= 3 and w*h < 4000*len(text.strip()):
                    # Translate the text using deep_translator
                    try:
                        translated = self.translator.translate_text(text, target_lang=target_lang).text
                    except Exception as e:
                        print(f"Translation error for text '{text}': {e}")
                        translated = text  # Fallback to original text
                    
                    # Save the translation info
                    translations.append({
                        'region': (x, y, w, h),
                        'original': text,
                        'translated': translated
                    })
                    
                bg_padding = 0
                bg_x1 = max(0, x - bg_padding)
                bg_y1 = max(0, y - bg_padding)
                bg_x2 = x + w + bg_padding
                bg_y2 = y + h + bg_padding
                
                # Draw rounded rectangle (requires Pillow >= 8.0.0)
                try:
                    draw.rounded_rectangle(
                        (bg_x1, bg_y1, bg_x2, bg_y2),
                        fill='white',
                        width=1,
                        
                    )
                except AttributeError:
                    # Fallback to normal rectangle if rounded_rectangle not available
                    draw.rectangle((bg_x1, bg_y1, bg_x2, bg_y2), fill='white', outline='black', width=1)

                # 2. Font handling with better fallback
                try:
                    font = ImageFont.truetype(self.font_path, self.font_size)
                except (IOError, AttributeError):
                    try:
                        # Try common system fonts as fallback
                        for fallback_font in ['Arial.ttf', 'DejaVuSans.ttf', 'LiberationSans-Regular.ttf']:
                            try:
                                font = ImageFont.truetype(fallback_font, self.font_size)
                                break
                            except IOError:
                                continue
                        else:
                            font = ImageFont.load_default()
                    except:
                        font = ImageFont.load_default()

                # 3. Improved text wrapping with text length estimation
                avg_char_width = font.getlength("M")  # Better width estimation
                max_chars_per_line = max(2, int(w // avg_char_width)) if avg_char_width > 0 else 20
               
                wrapped_text = self._wrap_text(translated, max_chars_per_line)
                
                # 4. Better text positioning and alignment
                line_height = font.size + 2
                total_text_height = len(wrapped_text) * line_height
                text_y = y + (h - total_text_height) // 2  # Vertical centering
                
                # 5. Draw each line with proper alignment and shadow effect
                for line in wrapped_text:
                    text_width = font.getlength(line)
                    
                    # Calculate x position for centered alignment
                    text_x = x + (w - text_width) // 2
                    
                    # Text shadow (optional)
                    shadow_offset = 1
                    draw.text(
                        (text_x + shadow_offset, text_y + shadow_offset),
                        line,
                        fill='#888888',  # Shadow color
                        font=font
                    )
                    
                    # Main text
                    draw.text(
                        (text_x, text_y),
                        line,
                        fill='black',  # Main text color
                        font=font
                    )
                    
                    text_y += line_height
              
            
            except Exception as e:
                print(f"Error processing region {i}: {e}")
        
        return result_img
    
    def _wrap_text(self, text, max_chars_per_line, font=None):
        """Improved text wrapping function that considers font metrics"""
        if not text:
            return []
        
        try:
            # Try to get the font if not provided
            if font is None:
                font = ImageFont.truetype(self.font_path, self.font_size) if hasattr(self, 'font_path') else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        words = text.split()
        if not words:
            return []
        
        lines = []
        current_line = []
        current_line_width = 0
        
        # Approximate average character width for fallback calculation
        avg_char_width = font.getlength("M") if hasattr(font, 'getlength') else self.font_size / 2
        
        space_width = font.getlength(" ") if hasattr(font, 'getlength') else avg_char_width
        
        for word in words:
            word_width = font.getlength(word) if hasattr(font, 'getlength') else len(word) * avg_char_width
            
            # Check if word fits in current line (considering space if not first word)
            if current_line:
                test_width = current_line_width + space_width + word_width
            else:
                test_width = word_width
            
            if test_width <= max_chars_per_line * avg_char_width:
                current_line.append(word)
                current_line_width = test_width
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_line_width = word_width
                else:
                    # Handle case where a single word is too long
                    lines.append(word)
                    current_line = []
                    current_line_width = 0
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    def translate_manga_page(self, image, target_lang='en'):
        """Main function to translate a manga page"""
        # Detect text regions
        regions = self.detect_text_regions(image)
        
        # Convert boxes from [x1, y1, x2, y2] to (x, y, w, h) format
        boxes = self.convert_to_xywh(regions['boxes'])
    

        return self.recognize_and_translate(image, boxes, target_lang)
        
        

    


        