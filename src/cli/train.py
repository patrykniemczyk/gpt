"""Command line interface for training GPT models."""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gpt.config import load_config, GPTConfig
from gpt.model import GPT
from gpt.tokenizer import BPETokenizer
from gpt.training import Trainer
from gpt.training.dataset import load_text_data, prepare_datasets, create_dataloader, create_streaming_dataloader
from gpt.utils.logging import setup_logging, get_logger


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train a GPT model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--tokenizer-only", 
        action="store_true",
        help="Only train the tokenizer, don't train the model"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Run through setup without actual training"
    )
    parser.add_argument(
        "--use-streaming", 
        action="store_true",
        help="Use streaming data loading (recommended for large datasets)"
    )
    parser.add_argument(
        "--no-streaming", 
        action="store_true",
        help="Disable streaming data loading (load all data into memory)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config from {args.config}: {e}")
        return 1
    
    # Setup logging
    setup_logging(log_dir=config.files.log_dir, log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("Starting GPT training")
    logger.info(f"Configuration loaded from: {args.config}")
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Determine streaming mode
    use_streaming = not args.no_streaming
    if args.use_streaming:
        use_streaming = True
    
    logger.info(f"Streaming mode: {use_streaming}")
    
    try:
        # Prepare fallback texts in case dataset loading fails
        fallback_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning networks can learn complex patterns from data.",
            "Transformers have revolutionized natural language processing.",
            "Attention mechanisms help models focus on relevant information.",
            "Large language models can generate coherent and contextually relevant text.",
            "Tokenization is the process of breaking text into smaller units.",
            "Training neural networks requires large amounts of computational resources.",
            "Generative models can create new content based on learned patterns.",
            "Self-attention allows models to weigh the importance of different words.",
            "Backpropagation is used to train neural networks by adjusting weights.",
            "Gradient descent is an optimization algorithm used in machine learning.",
            "Embeddings represent words as dense vectors in high-dimensional space.",
            "Batch normalization helps stabilize training in deep neural networks.",
            "Dropout is a regularization technique that prevents overfitting.",
            "Recurrent neural networks can process sequential data.",
            "Convolutional neural networks excel at image recognition tasks.",
            "Transfer learning allows models to leverage pre-trained knowledge.",
            "Fine-tuning adapts pre-trained models to specific tasks.",
            "The field of artificial intelligence continues to evolve rapidly.",
            "Computer vision enables machines to interpret and understand images.",
            "Speech recognition converts spoken language into text.",
            "Reinforcement learning trains agents to make decisions through trial and error.",
            "Unsupervised learning discovers hidden patterns in unlabeled data.",
            "Supervised learning uses labeled examples to train predictive models.",
            "Data preprocessing is crucial for successful machine learning projects.",
            "Feature engineering involves creating relevant input variables for models.",
            "Model evaluation measures how well algorithms perform on test data.",
            "Cross-validation helps assess model generalization performance.",
            "Hyperparameter tuning optimizes model configuration settings.",
            "Ensemble methods combine multiple models to improve predictions.",
            "Regularization techniques prevent models from overfitting training data.",
            "Activation functions introduce non-linearity in neural networks.",
            "Loss functions measure the difference between predictions and true values.",
            "Optimizers determine how model parameters are updated during training.",
            "Learning rate scheduling adjusts the step size during optimization.",
            "Batch size affects the stability and speed of training.",
            "Epochs represent complete passes through the training dataset.",
            "Validation sets help monitor model performance during training.",
            "Test sets provide unbiased evaluation of final model performance.",
            "Overfitting occurs when models memorize training data instead of learning patterns.",
            "Underfitting happens when models are too simple to capture data patterns.",
            "The bias-variance tradeoff is fundamental in machine learning.",
            "Dimensionality reduction techniques help manage high-dimensional data.",
            "Principal component analysis finds the most important data dimensions.",
            "Clustering algorithms group similar data points together.",
            "Classification predicts discrete categories or labels.",
            "Regression predicts continuous numerical values.",
            "Time series analysis handles data collected over time.",
            "Anomaly detection identifies unusual patterns in data.",
            "Recommendation systems suggest relevant items to users.",
            "Information retrieval finds relevant documents for user queries.",
            "Semantic similarity measures how closely related two pieces of text are.",
            "Named entity recognition identifies specific types of entities in text.",
            "Part-of-speech tagging labels words with their grammatical roles.",
            "Sentiment analysis determines the emotional tone of text.",
            "Topic modeling discovers themes in collections of documents.",
            "Text summarization creates concise versions of longer documents.",
            "Machine translation converts text from one language to another.",
            "Question answering systems provide responses to user queries.",
            "Dialogue systems engage in conversations with users.",
            "Chatbots simulate human conversation through text or voice.",
            "Virtual assistants help users complete tasks through natural language.",
            "Knowledge graphs represent information as networks of entities and relationships.",
            "Ontologies define concepts and relationships in specific domains.",
            "Semantic web technologies enable machines to understand web content.",
            "Linked data connects information across different sources.",
            "Data mining extracts useful patterns from large datasets.",
            "Big data analytics handles massive volumes of information.",
            "Distributed computing processes data across multiple machines.",
            "Cloud computing provides scalable resources for data processing.",
            "Edge computing brings computation closer to data sources.",
            "Internet of Things connects everyday objects to the internet.",
            "Blockchain technology ensures secure and transparent transactions.",
            "Cryptocurrency uses cryptographic techniques for digital currencies.",
            "Cybersecurity protects digital systems from threats and attacks.",
            "Privacy-preserving techniques protect sensitive information.",
            "Federated learning trains models without centralizing data.",
            "Differential privacy adds noise to protect individual privacy.",
            "Explainable AI makes machine learning decisions more interpretable.",
            "Fairness in AI ensures equitable treatment across different groups.",
            "Ethical AI considers the moral implications of artificial intelligence.",
            "Responsible AI development prioritizes safety and beneficial outcomes.",
            "Human-AI collaboration combines human expertise with machine capabilities.",
            "Augmented intelligence enhances human decision-making with AI assistance.",
            "Automation streamlines repetitive tasks and processes.",
            "Robotics combines AI with mechanical systems for physical tasks.",
            "Autonomous vehicles use AI for self-driving capabilities.",
            "Smart cities leverage technology to improve urban life.",
            "Precision agriculture uses AI to optimize farming practices.",
            "Personalized medicine tailors treatments to individual patients.",
            "Drug discovery accelerates with AI-assisted research.",
            "Medical imaging analysis helps diagnose diseases more accurately.",
            "Predictive maintenance prevents equipment failures before they occur.",
            "Supply chain optimization reduces costs and improves efficiency.",
            "Financial modeling predicts market trends and risks.",
            "Algorithmic trading uses AI to make investment decisions.",
            "Fraud detection identifies suspicious financial transactions.",
            "Customer service chatbots provide 24/7 support.",
            "Personalized recommendations increase user engagement.",
            "A/B testing compares different versions of products or services.",
            "User experience design focuses on creating intuitive interfaces.",
            "Human-computer interaction studies how people interact with technology.",
            "Accessibility ensures technology is usable by people with disabilities.",
            "Inclusive design considers diverse user needs and abilities.",
            "Sustainable technology minimizes environmental impact.",
            "Green computing reduces energy consumption in data centers.",
            "Quantum computing promises exponential speedups for certain problems.",
            "Neuromorphic computing mimics the structure of biological brains.",
            "Optical computing uses light instead of electricity for processing.",
            "DNA computing stores information in biological molecules.",
            "Biocomputing combines biology and computer science.",
            "Synthetic biology engineers biological systems for specific purposes.",
            "Nanotechnology manipulates matter at the atomic scale.",
            "Materials science develops new substances with desired properties.",
            "Renewable energy sources provide sustainable power generation.",
            "Energy storage systems enable efficient use of renewable energy.",
            "Smart grids optimize electricity distribution and consumption.",
            "Electric vehicles reduce dependence on fossil fuels.",
            "Sustainable transportation minimizes environmental impact.",
            "Climate modeling predicts future environmental conditions.",
            "Environmental monitoring tracks ecosystem health.",
            "Conservation efforts protect endangered species and habitats.",
            "Biodiversity research studies the variety of life on Earth.",
            "Ecology examines interactions between organisms and their environment.",
            "Evolutionary biology explains how species change over time.",
            "Genetics studies heredity and variation in living organisms.",
            "Genomics analyzes entire genomes of organisms.",
            "Proteomics studies the complete set of proteins in an organism.",
            "Bioinformatics applies computational methods to biological data.",
            "Systems biology takes a holistic approach to understanding life.",
            "Synthetic biology engineers biological systems for practical applications.",
            "Biotechnology uses living organisms to develop useful products.",
            "Pharmaceutical research develops new medications and treatments.",
            "Clinical trials test the safety and efficacy of new treatments.",
            "Epidemiology studies the distribution and causes of diseases.",
            "Public health focuses on preventing disease and promoting wellness.",
            "Global health addresses health issues that transcend national boundaries.",
            "One health recognizes connections between human, animal, and environmental health.",
            "Precision public health uses data to target interventions more effectively.",
            "Social determinants of health influence population health outcomes.",
            "Health equity ensures fair access to healthcare for all people.",
            "Mental health encompasses emotional, psychological, and social well-being.",
            "Preventive medicine focuses on avoiding disease before it occurs.",
            "Lifestyle medicine addresses the root causes of chronic diseases.",
            "Integrative medicine combines conventional and alternative approaches.",
            "Telemedicine provides remote healthcare services.",
            "Digital health uses technology to improve health outcomes.",
            "Wearable devices monitor health metrics continuously.",
            "Mobile health applications provide health services through smartphones.",
            "Electronic health records digitize patient information.",
            "Health information systems manage healthcare data.",
            "Interoperability allows different health systems to communicate.",
            "Health informatics applies information technology to healthcare.",
            "Evidence-based medicine uses research to guide clinical decisions.",
            "Translational research bridges laboratory discoveries and clinical practice.",
            "Collaborative research brings together experts from different fields.",
            "Open science promotes transparency and reproducibility in research.",
            "Citizen science engages the public in scientific research.",
            "Science communication makes research accessible to broader audiences.",
            "Science policy guides how scientific knowledge informs decision-making.",
            "Innovation drives progress in science and technology.",
            "Entrepreneurship creates new businesses and solutions.",
            "Technology transfer moves innovations from research to market.",
            "Intellectual property protects innovations and creative works.",
            "Patent systems encourage innovation by granting exclusive rights.",
            "Copyright protects original works of authorship.",
            "Trademark law protects brand names and logos.",
            "Trade secrets protect confidential business information.",
            "Licensing agreements allow others to use intellectual property.",
            "Technology licensing enables access to patented innovations.",
            "Research and development drives technological advancement.",
            "Basic research expands fundamental knowledge.",
            "Applied research addresses specific practical problems.",
            "Experimental design ensures reliable and valid results.",
            "Statistical analysis helps interpret research data.",
            "Meta-analysis combines results from multiple studies.",
            "Systematic reviews provide comprehensive summaries of research.",
            "Peer review ensures quality and rigor in scientific publications.",
            "Scientific method provides a systematic approach to understanding.",
            "Hypothesis testing evaluates proposed explanations.",
            "Replication confirms the reliability of research findings.",
            "Reproducibility ensures that results can be obtained consistently.",
            "Transparency promotes trust in scientific research.",
            "Scientific integrity maintains honesty and objectivity.",
            "Research ethics ensures responsible conduct of research.",
            "Informed consent protects research participants.",
            "Institutional review boards oversee research involving human subjects.",
            "Animal research ethics ensures humane treatment of research animals.",
            "Environmental ethics guides responsible interaction with nature.",
            "Sustainability balances current needs with future generations.",
            "Circular economy minimizes waste and maximizes resource use.",
            "Life cycle assessment evaluates environmental impacts.",
            "Carbon footprint measures greenhouse gas emissions.",
            "Climate change mitigation reduces causes of global warming.",
            "Climate adaptation adjusts to changing environmental conditions.",
            "Resilience enables systems to recover from disruptions.",
            "Disaster preparedness reduces vulnerability to emergencies.",
            "Risk assessment identifies potential hazards and their impacts.",
            "Risk management implements strategies to minimize adverse effects.",
            "Business continuity ensures operations can continue during disruptions.",
            "Crisis management responds effectively to unexpected events.",
            "Emergency response coordinates immediate actions during crises.",
            "Incident command systems organize emergency response efforts.",
            "Communication strategies disseminate information during emergencies.",
            "Community engagement involves stakeholders in decision-making.",
            "Stakeholder analysis identifies parties affected by decisions.",
            "Participatory approaches include communities in planning processes.",
            "Consensus building seeks agreement among diverse groups.",
            "Conflict resolution addresses disagreements constructively.",
            "Negotiation skills help reach mutually beneficial agreements.",
            "Mediation uses neutral third parties to facilitate discussions.",
            "Arbitration provides binding decisions to resolve disputes.",
            "Collaborative governance involves multiple stakeholders in decision-making.",
            "Public participation strengthens democratic processes.",
            "Civic engagement encourages active citizenship.",
            "Social capital builds networks of relationships among people.",
            "Community development improves quality of life in neighborhoods.",
            "Urban planning designs livable and sustainable cities.",
            "Transportation planning creates efficient mobility systems.",
            "Infrastructure investment supports economic growth.",
            "Economic development creates jobs and improves living standards.",
            "Sustainable development meets present needs without compromising the future.",
            "International cooperation addresses global challenges together.",
            "Diplomacy manages relationships between nations.",
            "Peacebuilding prevents and resolves conflicts.",
            "Humanitarian aid assists people in crisis situations.",
            "Development aid supports economic and social progress.",
            "Capacity building strengthens institutions and skills.",
            "Knowledge transfer shares expertise and best practices.",
            "Cultural exchange promotes understanding between different societies.",
            "Cross-cultural communication bridges differences between groups.",
            "Diversity and inclusion create welcoming environments for all.",
            "Equity ensures fair treatment and opportunities.",
            "Social justice addresses systemic inequalities.",
            "Human rights protect fundamental freedoms and dignity.",
            "Gender equality promotes equal opportunities for all genders.",
            "Youth empowerment develops skills and leadership in young people.",
            "Elder care supports the needs of aging populations.",
            "Family support strengthens relationships and well-being.",
            "Community health addresses population health at local levels.",
            "Social determinants influence health outcomes.",
            "Health promotion encourages healthy behaviors.",
            "Disease prevention reduces the burden of illness.",
            "Vaccination programs protect communities from infectious diseases.",
            "Antimicrobial resistance threatens the effectiveness of antibiotics.",
            "One health approaches recognize interconnections between human, animal, and environmental health.",
            "Zoonotic diseases can spread from animals to humans.",
            "Vector-borne diseases are transmitted by insects and other vectors.",
            "Waterborne diseases spread through contaminated water.",
            "Foodborne illnesses result from contaminated food.",
            "Air quality affects respiratory health.",
            "Noise pollution impacts hearing and well-being.",
            "Chemical exposure can cause health problems.",
            "Radiation safety protects against harmful exposure.",
            "Occupational health ensures safe working conditions.",
            "Ergonomics designs workspaces to prevent injury.",
            "Safety culture promotes awareness and prevention of hazards.",
            "Risk communication helps people understand and respond to risks.",
            "Emergency preparedness saves lives during disasters.",
            "First aid provides immediate care for injuries.",
            "Life support systems maintain vital functions.",
            "Trauma care treats severe injuries.",
            "Rehabilitation helps people recover from injuries or illnesses.",
            "Palliative care improves quality of life for seriously ill patients.",
            "Hospice care provides comfort and support at end of life.",
            "Grief counseling helps people cope with loss.",
            "Bereavement support assists families after death.",
            "Spiritual care addresses religious and existential needs.",
            "Holistic care considers the whole person.",
            "Person-centered care focuses on individual needs and preferences.",
            "Shared decision-making involves patients in treatment choices.",
            "Health literacy enables people to understand health information.",
            "Patient education improves health outcomes.",
            "Self-management helps people take control of their health.",
            "Chronic disease management addresses long-term conditions.",
            "Acute care treats immediate health problems.",
            "Primary care provides comprehensive first-contact healthcare.",
            "Specialist care addresses specific medical conditions.",
            "Multidisciplinary care involves teams of healthcare professionals.",
            "Interprofessional collaboration improves patient outcomes.",
            "Care coordination ensures seamless transitions between providers.",
            "Continuity of care maintains relationships over time.",
            "Quality improvement enhances healthcare delivery.",
            "Patient safety prevents harm during medical care.",
            "Infection control prevents the spread of diseases.",
            "Medication management ensures safe and effective drug use.",
            "Clinical guidelines provide evidence-based recommendations.",
            "Best practices represent the most effective approaches.",
            "Continuous improvement seeks ongoing enhancement.",
            "Innovation drives positive change.",
            "Creativity generates new ideas and solutions.",
            "Problem-solving addresses challenges systematically.",
            "Critical thinking evaluates information objectively.",
            "Decision-making chooses among available options.",
            "Strategic planning sets long-term direction.",
            "Goal setting establishes clear objectives.",
            "Performance measurement tracks progress toward goals.",
            "Feedback loops provide information for improvement.",
            "Learning organizations adapt and evolve continuously.",
            "Knowledge management captures and shares organizational learning.",
            "Best practice sharing spreads effective approaches.",
            "Mentorship develops skills and knowledge.",
            "Professional development enhances career growth.",
            "Lifelong learning supports continuous skill development.",
            "Competency-based education focuses on mastery of skills.",
            "Experiential learning learns through direct experience.",
            "Service learning combines education with community service.",
            "Inquiry-based learning encourages questions and exploration.",
            "Problem-based learning uses real-world challenges.",
            "Project-based learning creates tangible outcomes.",
            "Collaborative learning involves working together.",
            "Peer learning shares knowledge among equals.",
            "Self-directed learning empowers individuals to guide their education.",
            "Personalized learning adapts to individual needs."
        ]
        
        # Extend fallback texts to meet sample requirements
        while len(fallback_texts) < 50000:  # Ensure we have enough for larger configs
            fallback_texts.extend(fallback_texts)
        
        # Load training data for tokenizer training
        logger.info("Loading training data for tokenizer...")
        tokenizer_texts = load_text_data(
            dataset_name=config.data.dataset_name,
            dataset_config=config.data.dataset_config,
            split=config.data.dataset_split,
            num_samples=config.data.tokenizer_training_samples,
            cache_file=str(Path(config.files.output_dir) / "tokenizer_data.json"),
            use_streaming=use_streaming,
            fallback_texts=fallback_texts[:config.data.tokenizer_training_samples]
        )
        logger.info(f"Loaded {len(tokenizer_texts)} texts for tokenizer training")
        
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = BPETokenizer(
            vocab_size=config.tokenizer.vocab_size or config.model.vocab_size - len(config.tokenizer.special_tokens),
            special_tokens=config.tokenizer.special_tokens
        )
        
        # Train or load tokenizer
        tokenizer_path = Path(config.tokenizer.path)
        if tokenizer_path.exists():
            logger.info(f"Loading existing tokenizer from {tokenizer_path}")
            tokenizer.load(tokenizer_path)
        else:
            logger.info("Training tokenizer...")
            tokenizer.train(tokenizer_texts)
            logger.info(f"Saving tokenizer to {tokenizer_path}")
            tokenizer.save(tokenizer_path)
        
        logger.info(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
        
        # If tokenizer-only mode, exit here
        if args.tokenizer_only:
            logger.info("Tokenizer-only mode: training complete")
            return 0
        
        # Choose data loading strategy
        if use_streaming:
            logger.info("Using streaming data loading...")
            # Create streaming data loaders
            train_dataloader = create_streaming_dataloader(
                dataset_name=config.data.dataset_name,
                dataset_config=config.data.dataset_config,
                split=config.data.dataset_split,
                tokenizer=tokenizer,
                max_length=config.model.max_block_size,
                batch_size=config.training.batch_size,
                max_samples=config.data.num_training_samples,
                cache_file=str(Path(config.files.output_dir) / "train_data.json"),
                fallback_texts=fallback_texts[:config.data.num_training_samples]
            )
            
            # For streaming, we create a separate validation stream
            # by using a different split or subset
            eval_dataloader = None
            if config.data.validation_split > 0:
                # Use a portion of training data for validation
                val_samples = int(config.data.num_training_samples * config.data.validation_split)
                eval_dataloader = create_streaming_dataloader(
                    dataset_name=config.data.dataset_name,
                    dataset_config=config.data.dataset_config,
                    split=config.data.dataset_split,
                    tokenizer=tokenizer,
                    max_length=config.model.max_block_size,
                    batch_size=config.training.batch_size,
                    max_samples=val_samples,
                    cache_file=str(Path(config.files.output_dir) / "val_data.json"),
                    fallback_texts=fallback_texts[config.data.num_training_samples:config.data.num_training_samples + val_samples]
                )
        else:
            logger.info("Using traditional data loading...")
            # Load all training data
            texts = load_text_data(
                dataset_name=config.data.dataset_name,
                dataset_config=config.data.dataset_config,
                split=config.data.dataset_split,
                num_samples=config.data.num_training_samples,
                cache_file=config.data.data_file,
                use_streaming=False,
                fallback_texts=fallback_texts[:config.data.num_training_samples]
            )
            
            # Prepare datasets
            logger.info("Preparing datasets...")
            train_tokenized, val_tokenized = prepare_datasets(
                texts=texts,
                tokenizer=tokenizer,
                max_length=config.model.max_block_size,
                validation_split=config.data.validation_split
            )
            
            # Create data loaders
            train_dataloader = create_dataloader(
                tokenized_texts=train_tokenized,
                tokenizer=tokenizer,
                max_length=config.model.max_block_size,
                batch_size=config.training.batch_size,
                shuffle=True
            )
            
            eval_dataloader = None
            if val_tokenized:
                eval_dataloader = create_dataloader(
                    tokenized_texts=val_tokenized,
                    tokenizer=tokenizer,
                    max_length=config.model.max_block_size,
                    batch_size=config.training.batch_size,
                    shuffle=False
                )
        
        # Initialize model
        logger.info("Initializing model...")
        model = GPT(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=config.model.embed_dim,
            ff_dim=config.model.ff_dim,
            num_layers=config.model.num_layers,
            heads=config.model.heads,
            dropout=config.model.dropout,
            max_seq_len=config.model.max_block_size,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device
        )
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run complete - setup successful")
            return 0
        
        # Generate initial sample
        logger.info("Generating initial sample...")
        initial_sample = trainer.generate_sample(
            prompt="The future of artificial intelligence",
            max_new_tokens=50,
            temperature=config.sampling.temperature_default
        )
        logger.info(f"Initial sample: {initial_sample}")
        
        # Start training
        logger.info("Starting training...")
        results = trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            resume_from_checkpoint=args.resume
        )
        
        # Generate final sample
        logger.info("Generating final sample...")
        final_sample = trainer.generate_sample(
            prompt="The future of artificial intelligence",
            max_new_tokens=100,
            temperature=config.sampling.temperature_default
        )
        logger.info(f"Final sample: {final_sample}")
        
        # Log training results
        logger.info("Training completed successfully!")
        logger.info(f"Training results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
        return 1


if __name__ == "__main__":
    sys.exit(main())