import spacy
nlp = spacy.load("en_core_web_sm")
# filter for captions that contain both subject,predicate and adverbial
def find_predicate_and_subject(doc):
   
    predicate_token = None
    subject_token = None

    # Strategy 1: ROOT is a VERB (typical sentence)
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            predicate_token = token
            for child in predicate_token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject_token = child
                    return predicate_token, subject_token
            break 

    # Strategy 2: ROOT is a NOUN, look for an adjectival clause (acl) as predicate
    # This handles "A [noun] [verb-ing]..." structures like your examples
    if not predicate_token:
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "NOUN":
                subject_candidate = token # The ROOT noun is likely the subject
                for child in token.children:
                    if child.dep_ == "acl" and child.pos_ == "VERB": # acl: adjectival clause
                        predicate_token = child
                        subject_token = subject_candidate # Head of acl is the subject
                        return predicate_token, subject_token
                # If ROOT is NOUN but no acl VERB, it might be a noun phrase without a clear action verb
                # For this specific filter, we need a predicate verb.
                break # Processed ROOT noun

    # Strategy 3: If no clear structure, find first nsubj and its head verb
    if not predicate_token or not subject_token:
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                subject_token = token
                predicate_token = token.head
                return predicate_token, subject_token

    return predicate_token, subject_token


def has_attribute_for_subject(subject_token, doc):
    """
    Checks if the given subject_token has any adjectival modifiers (定语).
    """
    if not subject_token:
        return False
    for token in doc:
        # amod: adjectival modifier (e.g., "small" bird)
        # poss: possessive determiner (e.g., "its" mouth, if subject is "mouth")
        # compound: noun modifying noun (e.g. "apple" tree, if subject is "tree")
        # det: determiner (e.g. "A" bird, "The" man) - you might or might not want this.
        #      For "small bird", "small" is amod. "A" is det.
        #      Let's focus on amod, poss, compound for clear attributes.
        if token.head == subject_token:
            if token.dep_ in ("amod", "poss", "compound"):
                # print(f"DEBUG: Found attribute '{token.text}' for subject '{subject_token.text}'")
                return True
            # Prepositional phrase modifying the subject
            # e.g., "The cat [on the mat]" - "on the mat" modifies "cat"
            if token.dep_ == "prep":
                # Check if this prep phrase itself isn't the main adverbial for the predicate
                # This can get tricky, but for attributes, a prep phrase directly attached to subject is a candidate.
                # print(f"DEBUG: Found potential prep attribute '{token.text}' for subject '{subject_token.text}'")
                return True
    return False


def has_adverbial_for_predicate(predicate_token, doc):
    """
    Checks if the given predicate_token has any adverbial modifiers (状语).
    """
    if not predicate_token:
        return False
    for token in doc:
        if token.head == predicate_token:
            # advmod: adverbial modifier (e.g., "quickly" runs)
            # prep + pobj: prepositional phrase as adverbial (e.g., runs "in the park")
            if token.dep_ in ("advmod", "npadvmod"): # npadvmod for noun phrase as adverbial
                # print(f"DEBUG: Found adverbial '{token.text}' for predicate '{predicate_token.text}'")
                return True
            if token.dep_ == "prep": # Prepositional phrase modifying the predicate
                # print(f"DEBUG: Found prep adverbial '{token.text}' for predicate '{predicate_token.text}'")
                return True
        # advcl: adverbial clause modifier
        if token.dep_ == "advcl" and token.head == predicate_token:
            # print(f"DEBUG: Found advcl adverbial '{token.text}' for predicate '{predicate_token.text}'")
            return True
            
    return False


def analyze_and_filter_sentence(sentence_text):
    """
    Analyzes a sentence to see if it contains a subject, predicate,
    an attribute for the subject, and an adverbial for the predicate.
    """
    doc = nlp(sentence_text)
    
    predicate, subject = find_predicate_and_subject(doc)

    if not subject:
        # print("  => Condition unmet: No clear subject found.")
        return False
    if not predicate:
        # print("  => Condition unmet: No clear predicate found.")
        return False

    has_attr = has_attribute_for_subject(subject,doc)
    if not has_attr:
        return False

    has_adv = has_adverbial_for_predicate(predicate, doc)
    if not has_adv:
        # print(f"  => Condition unmet: Predicate '{predicate.text}' has no qualifying adverbial.")
        return False

    # print(f"  => All conditions met!")
    return True


def filter_sentences_from_list(sentence_list):
    """
    Filters a list of sentences, keeping only those that meet all criteria.
    """
    filtered_sentences = []
    print(f"Processing {len(sentence_list)} sentences...")
    for i, sentence in enumerate(sentence_list):
        if not sentence.strip(): # Skip empty sentences
            continue
        passes_filter = analyze_and_filter_sentence(sentence)
        if passes_filter:
            filtered_sentences.append(sentence)
            print(f"KEEPING ({i+1}): \"{sentence}\" (Reason: )")
        else:
            print(f"DROPPING ({i+1}): \"{sentence}\" (Reason: )")
            
    return filtered_sentences
