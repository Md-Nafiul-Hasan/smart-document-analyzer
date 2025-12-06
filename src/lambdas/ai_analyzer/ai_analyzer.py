import json
import re
from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class EntityExtractor:
    """Extract named entities and proper nouns from text"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'shall'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        try:
            sentences = sent_tokenize(text)
            entities = {
                'PERSON': [],
                'ORGANIZATION': [],
                'LOCATION': [],
                'PROPER_NOUNS': []
            }
            
            for sentence in sentences:
                # Use NLTK NE chunking for entity extraction
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                ne_tree = ne_chunk(pos_tags)
                
                # Extract named entities
                for subtree in ne_tree:
                    if hasattr(subtree, 'label'):
                        entity_name = ' '.join([word for word, tag in subtree.leaves()])
                        entity_type = subtree.label()
                        
                        if entity_type == 'PERSON' and entity_name not in entities['PERSON']:
                            entities['PERSON'].append(entity_name)
                        elif entity_type == 'ORGANIZATION' and entity_name not in entities['ORGANIZATION']:
                            entities['ORGANIZATION'].append(entity_name)
                        elif entity_type == 'GPE' and entity_name not in entities['LOCATION']:
                            entities['LOCATION'].append(entity_name)
                
                # Extract proper nouns as fallback
                proper_nouns = self._extract_proper_nouns(sentence)
                for noun in proper_nouns:
                    if noun not in entities['PROPER_NOUNS']:
                        entities['PROPER_NOUNS'].append(noun)
            
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {
                'PERSON': [],
                'ORGANIZATION': [],
                'LOCATION': [],
                'PROPER_NOUNS': []
            }
    
    def _extract_proper_nouns(self, sentence: str) -> List[str]:
        """Extract proper nouns from a sentence with better filtering"""
        try:
            tokens = word_tokenize(sentence)
            proper_nouns = []
            
            for i, word in enumerate(tokens):
                # Check for empty strings and single character words (with length check)
                if not word or len(word) < 2:
                    continue
                
                # Better length check before accessing word[0]
                if len(word) > 0 and word[0].isupper():
                    # Filter out common non-entity patterns
                    if (word not in self.stop_words and 
                        word not in self.common_words and
                        not word.startswith('"') and
                        not word.startswith("'") and
                        word.isalpha()):
                        
                        # Additional context-based filtering
                        if not self._is_sentence_start(i, tokens):
                            proper_nouns.append(word)
            
            return proper_nouns
        except Exception as e:
            logger.error(f"Error extracting proper nouns: {str(e)}")
            return []
    
    def _is_sentence_start(self, index: int, tokens: List[str]) -> bool:
        """Check if word is likely at sentence start (not a true proper noun)"""
        if index == 0:
            return True
        # Check if previous token is a sentence-ending punctuation
        if index > 0 and tokens[index - 1] in ['.', '!', '?']:
            return True
        return False


class SentimentAnalyzer:
    """Analyze sentiment of text"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the text"""
        try:
            if not text or len(text.strip()) == 0:
                return {
                    'overall_sentiment': 'NEUTRAL',
                    'confidence': 0.0,
                    'scores': {
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 1.0,
                        'compound': 0.0
                    },
                    'sentences': []
                }
            
            # Analyze overall sentiment
            overall_scores = self.sia.polarity_scores(text)
            
            # Determine sentiment label
            compound = overall_scores['compound']
            if compound >= 0.05:
                sentiment = 'POSITIVE'
            elif compound <= -0.05:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'NEUTRAL'
            
            # Analyze sentence-level sentiment
            sentences = sent_tokenize(text)
            sentence_sentiments = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 0:
                    scores = self.sia.polarity_scores(sentence)
                    sentence_compound = scores['compound']
                    
                    if sentence_compound >= 0.05:
                        sent_label = 'POSITIVE'
                    elif sentence_compound <= -0.05:
                        sent_label = 'NEGATIVE'
                    else:
                        sent_label = 'NEUTRAL'
                    
                    sentence_sentiments.append({
                        'sentence': sentence,
                        'sentiment': sent_label,
                        'confidence': abs(sentence_compound)
                    })
            
            return {
                'overall_sentiment': sentiment,
                'confidence': abs(compound),
                'scores': {
                    'positive': overall_scores['pos'],
                    'negative': overall_scores['neg'],
                    'neutral': overall_scores['neu'],
                    'compound': compound
                },
                'sentences': sentence_sentiments
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'overall_sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'scores': {
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'compound': 0.0
                },
                'sentences': []
            }


class KeywordExtractor:
    """Extract keywords from text"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.english_words = set(nltk.corpus.words.words())
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract keywords from text with improved filtering and scoring"""
        try:
            if not text or len(text.strip()) == 0:
                return []
            
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            
            # Filter tokens
            filtered_tokens = [
                token for token in tokens
                if (len(token) > 2 and  # Minimum length filter
                    token.isalpha() and  # Only alphabetic
                    token not in self.stop_words and
                    token in self.english_words)  # Valid English words
            ]
            
            if not filtered_tokens:
                return []
            
            # Calculate TF (Term Frequency)
            term_freq = {}
            for token in filtered_tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            # Calculate relevance score (TF with weighting)
            keywords = []
            total_tokens = len(filtered_tokens)
            
            for term, freq in term_freq.items():
                # TF score: frequency relative to total
                tf_score = freq / total_tokens
                
                # Boost score for longer terms (more specific)
                length_bonus = min(len(term) / 20, 0.3)
                
                # Final relevance score
                relevance_score = tf_score + length_bonus
                
                keywords.append({
                    'keyword': term,
                    'frequency': freq,
                    'relevance': round(relevance_score, 4)
                })
            
            # Sort by relevance and return top N
            keywords.sort(key=lambda x: x['relevance'], reverse=True)
            return keywords[:top_n]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []


class AIAnalyzer:
    """Main AI Analyzer class combining all analyzers"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Perform complete analysis on text"""
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Empty text provided for analysis")
                return {
                    'entities': {
                        'PERSON': [],
                        'ORGANIZATION': [],
                        'LOCATION': [],
                        'PROPER_NOUNS': []
                    },
                    'sentiment': {
                        'overall_sentiment': 'NEUTRAL',
                        'confidence': 0.0,
                        'scores': {
                            'positive': 0.0,
                            'negative': 0.0,
                            'neutral': 1.0,
                            'compound': 0.0
                        },
                        'sentences': []
                    },
                    'keywords': []
                }
            
            # Perform analyses
            entities = self.entity_extractor.extract_entities(text)
            sentiment = self.sentiment_analyzer.analyze_sentiment(text)
            keywords = self.keyword_extractor.extract_keywords(text)
            
            return {
                'entities': entities,
                'sentiment': sentiment,
                'keywords': keywords
            }
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler for AI analysis"""
    try:
        # Extract text from event
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event.get('body', {})
        text = body.get('text', '')
        
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No text provided'})
            }
        
        # Perform analysis
        analyzer = AIAnalyzer()
        result = analyzer.analyze(text)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
