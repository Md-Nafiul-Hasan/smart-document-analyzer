import json
import boto3
import logging
import chardet
from typing import Dict, Any, Optional, List
from pdf2image import convert_from_bytes
from PIL import Image
import io
import os

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
textract_client = boto3.client('textract')


class DocumentProcessor:
    """
    Process documents (PDF, images) and extract text using AWS Textract.
    Includes improved encoding detection and PDF validation.
    """
    
    # Supported document types
    SUPPORTED_FORMATS = {
        'application/pdf': 'pdf',
        'image/png': 'png',
        'image/jpeg': 'jpeg',
        'image/jpg': 'jpeg',
        'image/tiff': 'tiff'
    }
    
    # Maximum file size (25 MB for Textract)
    MAX_FILE_SIZE = 25 * 1024 * 1024
    
    def __init__(self):
        """Initialize the DocumentProcessor."""
        self.s3_client = s3_client
        self.textract_client = textract_client
    
    def validate_document(self, file_content: bytes, content_type: str) -> Dict[str, Any]:
        """
        Validate document format and content.
        
        Args:
            file_content: The document file content as bytes
            content_type: The MIME type of the document
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_size': len(file_content)
        }
        
        # Check file size
        if len(file_content) > self.MAX_FILE_SIZE:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"File size ({len(file_content)} bytes) exceeds maximum allowed ({self.MAX_FILE_SIZE} bytes)"
            )
        
        # Check file size minimum
        if len(file_content) < 100:
            validation_result['valid'] = False
            validation_result['errors'].append("File is too small or empty")
        
        # Validate format
        if content_type not in self.SUPPORTED_FORMATS:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Unsupported file format: {content_type}. Supported formats: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )
        
        # Additional validation for PDFs
        if content_type == 'application/pdf':
            if not self._validate_pdf(file_content):
                validation_result['valid'] = False
                validation_result['errors'].append("Invalid PDF file format")
        
        return validation_result
    
    def _validate_pdf(self, file_content: bytes) -> bool:
        """
        Validate PDF file structure.
        
        Args:
            file_content: The PDF file content as bytes
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            # Check PDF signature
            if not file_content.startswith(b'%PDF'):
                logger.warning("PDF file does not start with valid PDF signature")
                return False
            
            # Check for EOF marker
            if not b'%%EOF' in file_content:
                logger.warning("PDF file does not contain EOF marker")
                # This is a warning, not necessarily invalid - some PDFs might not have strict EOF
                return True
            
            return True
        except Exception as e:
            logger.error(f"Error validating PDF: {str(e)}")
            return False
    
    def detect_encoding(self, file_content: bytes) -> Dict[str, Any]:
        """
        Detect text encoding using chardet library with fallback strategies.
        
        Args:
            file_content: The file content as bytes
            
        Returns:
            Dictionary with detected encoding information
        """
        encoding_result = {
            'encoding': 'utf-8',
            'confidence': 0.0,
            'detection_method': 'default',
            'alternative_encodings': []
        }
        
        try:
            # Try chardet detection first
            detected = chardet.detect(file_content)
            
            if detected and detected.get('encoding'):
                encoding_result['encoding'] = detected['encoding']
                encoding_result['confidence'] = detected.get('confidence', 0.0)
                encoding_result['detection_method'] = 'chardet'
                
                # Validate detected encoding is valid
                try:
                    file_content.decode(detected['encoding'])
                except (UnicodeDecodeError, LookupError) as e:
                    logger.warning(f"Detected encoding {detected['encoding']} failed validation: {str(e)}")
                    encoding_result['encoding'] = 'utf-8'
                    encoding_result['confidence'] = 0.0
                    encoding_result['detection_method'] = 'fallback'
            else:
                logger.warning("Chardet could not detect encoding, using UTF-8")
                encoding_result['detection_method'] = 'fallback'
        
        except Exception as e:
            logger.warning(f"Error during encoding detection: {str(e)}. Using UTF-8.")
            encoding_result['detection_method'] = 'error_fallback'
        
        # Attempt to detect alternative encodings
        encoding_result['alternative_encodings'] = self._get_alternative_encodings(
            file_content, 
            encoding_result['encoding']
        )
        
        return encoding_result
    
    def _get_alternative_encodings(self, file_content: bytes, primary_encoding: str) -> List[str]:
        """
        Get alternative encodings that can decode the content.
        
        Args:
            file_content: The file content as bytes
            primary_encoding: The primary detected encoding
            
        Returns:
            List of alternative encodings that work
        """
        alternatives = []
        common_encodings = [
            'utf-8', 'utf-16', 'latin-1', 'iso-8859-1', 
            'cp1252', 'ascii', 'utf-16-le', 'utf-16-be'
        ]
        
        for encoding in common_encodings:
            if encoding.lower() == primary_encoding.lower():
                continue
            
            try:
                file_content.decode(encoding)
                alternatives.append(encoding)
            except (UnicodeDecodeError, LookupError):
                pass
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def process_pdf(self, file_content: bytes) -> Dict[str, Any]:
        """
        Process PDF file and extract text.
        
        Args:
            file_content: The PDF file content as bytes
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            logger.info("Processing PDF document")
            
            # Convert PDF to images
            images = convert_from_bytes(file_content, dpi=150)
            
            if not images:
                raise ValueError("PDF conversion resulted in no images")
            
            logger.info(f"Converted PDF to {len(images)} image(s)")
            
            extracted_text = ""
            document_metadata = {
                'total_pages': len(images),
                'format': 'pdf',
                'pages': []
            }
            
            # Process each page with Textract
            for page_num, image in enumerate(images, 1):
                try:
                    page_text = self._process_image_with_textract(image, page_num)
                    extracted_text += f"\n--- Page {page_num} ---\n{page_text}"
                    
                    document_metadata['pages'].append({
                        'page_number': page_num,
                        'status': 'success'
                    })
                except Exception as e:
                    logger.error(f"Error processing PDF page {page_num}: {str(e)}")
                    document_metadata['pages'].append({
                        'page_number': page_num,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return {
                'success': True,
                'text': extracted_text,
                'metadata': document_metadata
            }
        
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                'success': False,
                'error': f"PDF processing failed: {str(e)}",
                'text': ""
            }
    
    def process_image(self, file_content: bytes) -> Dict[str, Any]:
        """
        Process image file and extract text using Textract.
        
        Args:
            file_content: The image file content as bytes
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            logger.info("Processing image document")
            
            # Validate image
            try:
                image = Image.open(io.BytesIO(file_content))
                image.verify()
                # Need to reopen after verify
                image = Image.open(io.BytesIO(file_content))
            except Exception as e:
                raise ValueError(f"Invalid image file: {str(e)}")
            
            logger.info(f"Image format: {image.format}, Size: {image.size}")
            
            # Process with Textract
            text = self._process_image_with_textract(image)
            
            return {
                'success': True,
                'text': text,
                'metadata': {
                    'format': image.format or 'unknown',
                    'size': image.size,
                    'mode': image.mode
                }
            }
        
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                'success': False,
                'error': f"Image processing failed: {str(e)}",
                'text': ""
            }
    
    def _process_image_with_textract(self, image: Image.Image, page_num: Optional[int] = None) -> str:
        """
        Use AWS Textract to extract text from image.
        
        Args:
            image: PIL Image object
            page_num: Optional page number for logging
            
        Returns:
            Extracted text
        """
        try:
            # Convert PIL image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            page_info = f" (page {page_num})" if page_num else ""
            logger.info(f"Sending image{page_info} to Textract")
            
            # Call Textract
            response = self.textract_client.detect_document_text(
                Document={'Bytes': img_bytes.getvalue()}
            )
            
            # Extract text from response
            text = ""
            if 'Blocks' in response:
                for block in response['Blocks']:
                    if block['BlockType'] == 'LINE':
                        text += block.get('Text', '') + '\n'
            
            logger.info(f"Successfully extracted text{page_info}")
            return text
        
        except Exception as e:
            logger.error(f"Error in Textract processing: {str(e)}")
            raise
    
    def download_and_process(self, bucket: str, key: str, content_type: str) -> Dict[str, Any]:
        """
        Download document from S3 and process it.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            content_type: MIME type of the document
            
        Returns:
            Processing result
        """
        try:
            logger.info(f"Downloading {key} from {bucket}")
            
            # Download from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            file_content = response['Body'].read()
            
            # Validate document
            validation = self.validate_document(file_content, content_type)
            if not validation['valid']:
                logger.error(f"Document validation failed: {validation['errors']}")
                return {
                    'success': False,
                    'error': f"Document validation failed: {', '.join(validation['errors'])}",
                    'validation': validation
                }
            
            # Detect encoding for text files
            encoding_info = self.detect_encoding(file_content)
            logger.info(f"Detected encoding: {encoding_info['encoding']} (confidence: {encoding_info['confidence']})")
            
            # Process based on format
            if content_type == 'application/pdf':
                result = self.process_pdf(file_content)
            elif content_type in ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff']:
                result = self.process_image(file_content)
            else:
                return {
                    'success': False,
                    'error': f"Unsupported file format: {content_type}"
                }
            
            # Add encoding info to result
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata']['encoding'] = encoding_info
            
            return result
        
        except Exception as e:
            logger.error(f"Error in download_and_process: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def lambda_handler(event, context):
    """
    AWS Lambda handler for document processing.
    
    Expected event format:
    {
        'bucket': 'bucket-name',
        'key': 'document-key',
        'content_type': 'application/pdf'  # or image type
    }
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Extract parameters
        bucket = event.get('bucket')
        key = event.get('key')
        content_type = event.get('content_type', 'application/pdf')
        
        if not bucket or not key:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing required parameters: bucket and key'})
            }
        
        # Process document
        processor = DocumentProcessor()
        result = processor.download_and_process(bucket, key, content_type)
        
        status_code = 200 if result.get('success') else 400
        
        return {
            'statusCode': status_code,
            'body': json.dumps(result)
        }
    
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"Internal server error: {str(e)}"})
        }
