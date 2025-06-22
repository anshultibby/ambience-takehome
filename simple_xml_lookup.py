import xml.etree.ElementTree as ET
import re

def lookup_alphabetical_index(query, xml_file_path=None, xml_string=None) -> str:
    """
    Simple function to find main terms in alphabetical index and return in PDF-like format
    
    Args:
        query (str): The main medical term you're searching for
        xml_file_path (str): Path to your XML file (optional)
        xml_string (str): XML content as string (optional)
    
    Returns:
        list: List of formatted strings in PDF-like format
    """
    
    # Load XML - either from file or string
    if xml_file_path:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    elif xml_string:
        root = ET.fromstring(xml_string)
    else:
        raise ValueError("Either xml_file_path or xml_string must be provided")
    
    results = []
    query_lower = query.lower()
    
    def format_term(element, level=0):
        """Format a term element into PDF-like string format"""
        lines = []
        
        # Get title
        title_elem = element.find('title')
        if title_elem is None or not title_elem.text:
            return lines
            
        title = title_elem.text.strip()
        
        # Get code
        code_elem = element.find('code')
        code = code_elem.text.strip() if code_elem is not None and code_elem.text else None
        
        # Get cross-references
        see_elem = element.find('see')
        see = see_elem.text.strip() if see_elem is not None and see_elem.text else None
        
        seeAlso_elem = element.find('seeAlso')
        seeAlso = seeAlso_elem.text.strip() if seeAlso_elem is not None and seeAlso_elem.text else None
        
        # Create the formatted line
        indent = "- " * level if level > 0 else ""
        
        if code:
            line = f"{indent}{title} {code}"
        elif see:
            line = f"{indent}{title} -see {see}"
        elif seeAlso:
            line = f"{indent}{title} -see also {seeAlso}"
        else:
            line = f"{indent}{title}"
        
        lines.append(line)
        
        # Process nested terms
        for term in element.findall('term'):
            nested_lines = format_term(term, level + 1)
            lines.extend(nested_lines)
        
        return lines
    
    # Search through all mainTerm elements
    for main_term in root.findall('.//mainTerm'):
        title_elem = main_term.find('title')
        if title_elem is not None and title_elem.text:
            main_title = title_elem.text.strip().lower()
            
            # Check if this main term matches our query
            if query_lower in main_title:
                # Format this main term and all its sub-terms
                formatted_lines = format_term(main_term, 0)
                results.extend(formatted_lines)
    
    return results

def lookup_tabular_list(code, xml_file_path=None, xml_string=None) -> str:
    """
    Function to lookup ICD codes from XML tabular list and return easy-to-read text
    
    Args:
        code (str): The ICD code you're searching for (e.g., "A00", "E11.9")
        xml_file_path (str): Path to your tabular XML file (optional)
        xml_string (str): XML content as string (optional)
    
    Returns:
        str: Easy-to-read text format with code details
    """
    
    # Load XML - either from file or string
    if xml_file_path:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    elif xml_string:
        root = ET.fromstring(xml_string)
    else:
        raise ValueError("Either xml_file_path or xml_string must be provided")
    
    code_upper = code.upper().strip()
    
    # Find the specific code in the tabular list
    for diag in root.findall('.//diag'):
        name_elem = diag.find('name')
        if name_elem is not None and name_elem.text:
            diag_code = name_elem.text.strip()
            
            # Check for exact match or partial match (for codes with dashes)
            if diag_code == code_upper or diag_code.replace('-', '') == code_upper.replace('-', ''):
                
                # Get code details
                desc_elem = diag.find('desc')
                description = desc_elem.text.strip() if desc_elem is not None else ''
                
                # Find hierarchical structure
                chapter_info = None
                section_info = None
                
                # Find which section this code belongs to
                for section in root.findall('.//section'):
                    if diag in section.iter('diag'):
                        section_id = section.get('id', '')
                        desc_elem = section.find('desc')
                        section_info = {
                            'id': section_id,
                            'description': desc_elem.text.strip() if desc_elem is not None else ''
                        }
                        
                        # Find which chapter this section belongs to
                        for chapter in root.findall('.//chapter'):
                            if section in chapter.iter('section'):
                                name_elem = chapter.find('name')
                                desc_elem = chapter.find('desc')
                                chapter_info = {
                                    'number': name_elem.text.strip() if name_elem is not None else '',
                                    'description': desc_elem.text.strip() if desc_elem is not None else ''
                                }
                                break
                        break
                
                # Build the easy-to-read text output
                output_lines = []
                
                # Header with code and description
                output_lines.append(f"Code: {diag_code}")
                output_lines.append(f"Description: {description}")
                output_lines.append("")
                
                # Hierarchy
                if chapter_info:
                    output_lines.append(f"Chapter {chapter_info['number']}: {chapter_info['description']}")
                if section_info:
                    output_lines.append(f"Section {section_info['id']}: {section_info['description']}")
                output_lines.append("")
                
                # Includes
                includes_list = []
                for includes in diag.findall('.//includes/note'):
                    if includes.text:
                        includes_list.append(includes.text.strip())
                
                if includes_list:
                    output_lines.append("Includes:")
                    for item in includes_list:
                        output_lines.append(f"  - {item}")
                    output_lines.append("")
                
                # Excludes1
                excludes1_list = []
                for excludes1 in diag.findall('.//excludes1/note'):
                    if excludes1.text:
                        excludes1_list.append(excludes1.text.strip())
                
                if excludes1_list:
                    output_lines.append("Excludes1:")
                    for item in excludes1_list:
                        output_lines.append(f"  - {item}")
                    output_lines.append("")
                
                # Excludes2
                excludes2_list = []
                for excludes2 in diag.findall('.//excludes2/note'):
                    if excludes2.text:
                        excludes2_list.append(excludes2.text.strip())
                
                if excludes2_list:
                    output_lines.append("Excludes2:")
                    for item in excludes2_list:
                        output_lines.append(f"  - {item}")
                    output_lines.append("")
                
                # Inclusion terms
                inclusion_terms = []
                for inclusion in diag.findall('.//inclusionTerm/note'):
                    if inclusion.text:
                        inclusion_terms.append(inclusion.text.strip())
                
                if inclusion_terms:
                    output_lines.append("Inclusion Terms:")
                    for item in inclusion_terms:
                        output_lines.append(f"  - {item}")
                    output_lines.append("")
                
                # Use additional code
                use_additional = []
                for use_add in diag.findall('.//useAdditionalCode/note'):
                    if use_add.text:
                        use_additional.append(use_add.text.strip())
                
                if use_additional:
                    output_lines.append("Use Additional Code:")
                    for item in use_additional:
                        output_lines.append(f"  - {item}")
                    output_lines.append("")
                
                # Code first
                code_first = []
                for cf in diag.findall('.//codeFirst/note'):
                    if cf.text:
                        code_first.append(cf.text.strip())
                
                if code_first:
                    output_lines.append("Code First:")
                    for item in code_first:
                        output_lines.append(f"  - {item}")
                    output_lines.append("")
                
                # Code also
                code_also = []
                for ca in diag.findall('.//codeAlso/note'):
                    if ca.text:
                        code_also.append(ca.text.strip())
                
                if code_also:
                    output_lines.append("Code Also:")
                    for item in code_also:
                        output_lines.append(f"  - {item}")
                    output_lines.append("")
                
                return "\n".join(output_lines).strip()
    
    return f"Code {code} not found in tabular list."

def get_chapter_structure(xml_file_path=None, xml_string=None):
    """
    Function to get the complete chapter and section structure from tabular list
    
    Returns:
        list: List of chapters with their sections
    """
    
    # Load XML - either from file or string
    if xml_file_path:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    elif xml_string:
        root = ET.fromstring(xml_string)
    else:
        raise ValueError("Either xml_file_path or xml_string must be provided")
    
    chapters = []
    
    for chapter in root.findall('.//chapter'):
        name_elem = chapter.find('name')
        desc_elem = chapter.find('desc')
        
        chapter_data = {
            'number': name_elem.text.strip() if name_elem is not None else '',
            'description': desc_elem.text.strip() if desc_elem is not None else '',
            'sections': []
        }
        
        # Get section references from sectionIndex
        section_index = chapter.find('sectionIndex')
        if section_index is not None:
            for section_ref in section_index.findall('sectionRef'):
                section_data = {
                    'id': section_ref.get('id', ''),
                    'first': section_ref.get('first', ''),
                    'last': section_ref.get('last', ''),
                    'description': section_ref.text.strip() if section_ref.text else ''
                }
                chapter_data['sections'].append(section_data)
        
        chapters.append(chapter_data)
    
    return chapters

# Simplified version that just searches all text content
def simple_xml_search(query, xml_file_path=None, xml_string=None):
    """
    Even simpler search - just find all elements containing the query text
    """
    if xml_file_path:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    elif xml_string:
        root = ET.fromstring(xml_string)
    else:
        raise ValueError("Either xml_file_path or xml_string must be provided")
    
    matches = []
    query_lower = query.lower()
    
    for elem in root.iter():
        if elem.text and query_lower in elem.text.lower():
            matches.append({
                'tag': elem.tag,
                'text': elem.text.strip(),
                'attributes': dict(elem.attrib) if elem.attrib else {}
            })
    
    return matches

# Example usage:
if __name__ == "__main__":
    # Proper ICD-10 Coding Workflow
    print("=== PDF-like Format ICD-10 Lookup ===")
    
    # Step 1: Search for main term in Alphabetical Index
    print("Step 1: Search 'diabetes' in Alphabetical Index")
    print("-" * 50)
    alphabetical_index_path = "icd10cm_index_2025.xml" 
    alpha_results = lookup_alphabetical_index("diabetes", xml_file_path=alphabetical_index_path)
    
    # Show first 20 lines of the formatted output
    for line in alpha_results[:20]:
        print(line)
    
    print("\n... (showing first 20 lines)")
    print(f"Total lines: {len(alpha_results)}")
    
    # Step 2: Extract codes and look one up in tabular
    print(f"\nStep 2: Extract codes and look up in Tabular List")
    print("-" * 50)
    
    # Find first code in the results
    first_code = None
    for line in alpha_results:
        # Look for lines that end with a code pattern (letter followed by numbers)
        code_match = re.search(r'\b([A-Z]\d+(?:\.\d+)*)\b$', line)
        if code_match:
            first_code = code_match.group(1)
            print(f"Found code: {first_code} in line: {line}")
            break
    
    if first_code:
        tabular_list_path = "icd10cm_tabular_2025.xml"
        tabular_result = lookup_tabular_list(first_code, xml_file_path=tabular_list_path)
        
        if tabular_result:
            print(f"\nTabular details for {first_code}:")
            print(tabular_result)
    
    print(f"\nStep 3: Example of how to use with LLM")
    print("-" * 50)
    print("1. Pass the formatted alphabetical results to an LLM")
    print("2. LLM picks the most relevant code based on patient symptoms")
    print("3. Use lookup_tabular_list() to get complete code details")
    print("4. Get full hierarchy and coding instructions") 