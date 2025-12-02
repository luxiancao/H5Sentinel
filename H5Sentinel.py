import streamlit as st
import pandas as pd
from collections import defaultdict
from Bio import SeqIO, Align
from Bio.Align import substitution_matrices
import io
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. File Parsing ---
EXPECTED_SEGMENTS = {"PB2", "PB1", "PA", "HA", "NP", "NA", "M1", "NS1"}

def parse_fasta_file(uploaded_file):
    viruses = defaultdict(dict)
    parsing_errors = []
    try:
        decoded_file = uploaded_file.getvalue().decode("utf-8")
        fasta_io = io.StringIO(decoded_file)
    except Exception as e:
        parsing_errors.append(f"File read or decoding failed: {e}")
        return {}, [], parsing_errors
    for record in SeqIO.parse(fasta_io, "fasta"):
        header = record.id
        if "|" not in header:
            parsing_errors.append(f"Format Error: Sequence '{header}' missing '|' separator, skipped.")
            continue
        parts = header.split("|")
        virus_name = "|".join(parts[:-1])
        segment_name = parts[-1].upper()
        if not virus_name:
            parsing_errors.append(f"Format Error: Sequence '{header}' missing virus name before '|', skipped.")
            continue
        if segment_name in EXPECTED_SEGMENTS:
            if segment_name in viruses[virus_name]:
                parsing_errors.append(f"Warning: Virus '{virus_name}' has duplicate '{segment_name}' segment, using the last found sequence.")
            viruses[virus_name][segment_name] = str(record.seq)
        else:
            parsing_errors.append(f"Warning: Segment name '{segment_name}' for virus '{virus_name}' not recognized, skipped.")
    complete_viruses = {}
    incomplete_viruses = []
    for name, segments in viruses.items():
        missing = EXPECTED_SEGMENTS - segments.keys()
        if not missing:
            complete_viruses[name] = segments
        else:
            incomplete_viruses.append((name, missing))
    return complete_viruses, incomplete_viruses, parsing_errors

# --- 2. Model Loading ---
@st.cache_resource
def load_resources():
    """
    Load Model and Scaler and cache them.
    """
    resources = {}
    try:
        resources['model'] = joblib.load('rf_spillover_model.joblib')
        resources['scaler'] = joblib.load('scaler.joblib')
        return resources
    except FileNotFoundError as e:
        st.error(f"Critical file missing: {e.filename}. Please ensure 'rf_spillover_model.joblib' and 'scaler.joblib' are in the current directory.")
        return None
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None

# --- 3. Feature Extraction ---
SITES_OF_INTEREST = {
    "PB2": [299, 451, 480, 570, 627, 676, 699],
    "PB1": [215, 430, 456, 591, 694],
    "PA":  [86, 100, 184, 237, 602, 615, 716],
    "HA":  [2, 3, 5, 9, 10, 11, 12, 88, 99, 102, 104, 110, 131, 136, 142, 154, 
            170, 171, 172, 185, 211, 226, 285, 325, 390, 503, 506],
    "NP":  [52, 61, 186, 194, 353, 377, 450],
    "NA":  [270],
    "M1":   [95, 227],
    "NS1":  [7, 75, 84, 86, 138, 143, 197, 209, 215, 229]
}

AA_FEATURES_RAW = {
    'A': [-0.591, -1.302, -0.733, 1.57, -0.146],
    'C': [-1.343, 0.465, -0.862, -1.02, -0.255],
    'D': [1.05, 0.302, 3.656, 0.945, -0.371],
    'E': [1.357, -1.453, 1.477, 0.567, -0.177],
    'F': [-1.006, -0.590, 1.891, -1.46, 0.412],
    'G': [-0.384, 1.652, 1.330, 1.044, 0.189],
    'H': [0.336, -0.417, -0.544, 0.567, 0.023],
    'I': [-1.239, -0.547, -1.337, -0.861, 0.304],
    'K': [1.831, -0.561, 0.533, -0.277, -0.296],
    'L': [-1.019, -0.987, -1.505, -0.511, 0.212],
    'M': [-0.663, -1.524, 0.279, -0.911, 0.195],
    'N': [0.945, 0.828, 1.299, -0.179, 0.105],
    'P': [0.189, 2.081, -1.628, 0.421, 0.282],
    'Q': [0.931, -0.179, -0.425, 0.044, -0.091],
    'R': [1.538, -0.055, 1.502, 0.440, -0.375],
    'S': [-0.228, 0.420, -0.591, 0.660, 0.188],
    'T': [-0.032, 0.666, -1.333, 0.325, 0.291],
    'V': [-1.337, -0.279, -1.622, -0.912, 0.243],
    'W': [-0.595, 0.009, 0.672, -2.128, 0.675],
    'Y': [0.260, 0.830, 0.259, 0.051, 0.296],
    '-': [-0.13, -0.117, 0.6025, -0.028, -0.112],
    '?': [0, 0, 0, 0, 0],
    '*': [0, 0, 0, 0, 0],
    'X': [0, 0, 0, 0, 0]
}

def get_summed_features():
    return {aa: round(sum(values), 2) for aa, values in AA_FEATURES_RAW.items()}
SUMMED_FEATURES = get_summed_features()

REF_SEQUENCES = {
    "PB2": "MERIKELRDLMSQSRTREILTKTTVDHMAIIKKCTSGRQEKNPALRMKWMMAMKYPITADKRIMEMIPERNEQGQTLWSKTNDAGSDRVMVSPLAVTWWNRNGPTTSTVHYPKVYKTYFEKVERLKHGTFGPVHFRNQVKIRRRADINPGHADPSAKEAQDVIMEVVFPNEVGARILTSESQLTITKEKKEELQDCKIAPLMVAYMLERELVRKTRFLPVAGGTSSVYIEVLHLTQGTCWEQMYTPGGEVRNDDVDQSLIIAARNIVRRATVSADPQASLLEMCHSTQIGGIRMVDILRQNPTEEQAVDICKAAMGLRISSSFSFGGFTFKRTSGSSVKKEEEVLTGNLQTLKIRVHEGYEEFTMVGRRATAILRKATRRLIQLIVSGRDEQSIAEAIIVAMVFSQEDCMIKAVRGDLNFVNRANQRLNPMHQLLRHFQKDAKVLFQNWGIEPIDNVMGMIGILPDMTPSAEMSLRGVRVSKMGVDEYSSTERVVVSIDRFLRVRDQQGNVLLSPEEVSETQGTEKLTITYSSSMMWEINGPESVLVNTYQWIIRNWETVKIQWSQDPTMLYNKMEFESFQSLVPKAARGQYSGFVRTLFQQMRDVLGTFDTVQIIKLLPFAAAPPEQSRMQFSSLTVNVRGSGMRILVRGNSPVFNYNKATKRLTVLGKDAGALTEDPDEGTAGVESAVLRGFLILGREDKRYGPALSINELSNLAKGEKANVLIGQGDVVLVMKRKRDSSILTDSQTATKRIRMAIN",
    "PB1": "MDVNPTLLFLKVPAQNAISTTFPYTGDPPYSHGTGTGYTMDTVNRTHQYSEKGKWTTNTETGAPQLNPIDGPLPEDNEPSGYAQTDCVLEAMAFLEKSHPGIFENSCLETMEIVQQTRVDKLTQGRQTYDWTLNRNQPAATALANTIGVFRSNGLTANESGRLIDFLKDVMESMDKGEMEIITHFQRKRRVRDNMTKKMVTQRTIGKKKQRLNKRSYLIRALTLNTMTKDAERGKLKRRAIATPGMQIRGFVYFVETLARSICEKLEQSGLPVGGNEKKAKLANVVRKMMTNSQDTELSFTITGDNTKWNENQNPRMFLAMITYITRNQPEWFRNVLSIAPIMFSNKMARLGKGYMFESKSMKLRTQIPAEMLASIDLKYFNESTRKKIEKIRPLLIDGTASLSPGMMMGMFNMLSTVLGVSILNLGQKRYTKTTYWWDGLQSSDDFALIVNAPNHEGIQAGVDRFYRTCKLVGINMSKKKSYINRTGTFEFTSFFYRYGFVANFSMELPSFGVSGINESADMSIGVTVIKNNMINNDLGPATAQMALQLFIKDYRYTYRCHRGDTQIQTRRSFELKKLWEQTRSKAGLLVSDGGPNLYNIRNLHIPEVCLKWELMDEDYQGRLCNPLNPFVSHKEIESVNNAVVMPAHGPAKSMEYDAVATTHSWIPKRNRSILNTSQRGILEDEQMYQKCCNLFEKFFPSSSYRRPVGISSMVEAMVSRARIDARIDFESGRIKKEEFAEIMKICSTIEELRRQK",
    "PA":  "MEDFVRQCFNPMIVELAEKAMKEYGEDPKIETNKFAAICTHLEVCFMYSDFHFIDERGESTIIESGDPNALLKHRFEIIEGRDRTMAWTVVNSICNTTGIEKPKFLPDLYDYKENRFIEIGVTRREVHTYYLEKANKIKSEKTHIHIFSFTGEEMATKADYTLDEESRARIKTRLFTIRQEMASRGLWDSFRQSERGEETIEERFEITGTMCRLADQSLPPNFSSLEKFRAYVDGFEPNGCIEGKLSQMSKEVNARIEPFLKTTPRPLRLPDGPPCSQRSKFLLMDALKLSIEDPSHEGEGIPLYDAIKCMKTFFGWKEPNIVKPHEKGINPNYLLAWKQVLAELQDIENEEKIPKTKNMRKTSQLKWALGENMAPEKVDFEDCKDVSDLRQYDSDEPKPRSLASWIQSEFNKACELTDSSWIELDEIGEDVAPIEHIASMRRNYFTAEVSHCRATEYIMKGVYINTALLNASCAAMDDFQLIPMISKCRTKEGRRKTNLYGFIIKGRSHLRNDTDVVNFVSMEFSLTDPRLEPHKWEKYCVLEIGDMLLRTAIGQVSRPMFLYVRANGTSKIKMKWGMEMRRCLLQSLQQIESMIEAESSVKEKDMTKEFFENKSETWPIGESPKGMEEGSIGKVCRTLLAKSVFNSLYASPQLEGFSAESRKLLLIVQALRDNLEPGTFDLGGLYEAIEECLINDPWVLLNASWFNSFLTHALK",
    "HA":  "MEKIVLLLAIVSLVKSDQICIGYHANNSTEQVDTIMEKNVTVTHAQDILEKTHNGKLCDLNGVKPLILRDCSVAGWLLGNPMCDEFINVPEWSYIVEKASPANDLCYPGDFNDYEELKHLLSRTNHFEKIQIIPKSSWSNHDASSGVSSACPYHGRSSFFRNVVWLIKKNSAYPTIKRSYNNTNQEDLLVLWGIHHPNDAAEQTKLYQNPTTYISVGTSTLNQRLVPEIATRPKVNGQSGRMEFFWTILKPNDAINFESNGNFIAPEYAYKIVKKGDSAIMKSELEYGNCNTKCQTPMGAINSSMPFHNIHPLTIGECPKYVKSNRLVLATGLRNTPQREKRRKRGLFGAIAGFIEGGWQGMVDGWYGYRHSNEQGSGYAADKESTQKAIDGVTNKVNSIIDKMNTQFEAVGREFNNLERRIENLNKQMEDGLLDVWTYNAELLVLMENERTLDFHDSNVKNLYDKVRLQLRDNAKELGNGCFEFYHKCDNECMESVKNGTYDYPQYSEEARLNREEISGVKLESMGTYQILSIYSTVASSLALAIMVAGLSLWMCSNGSLQCRICI",
    "NP":  "MASQGTKRSYEQMETGGERQNATEIRASVGRMVGGVGRFYIQMCTELKLSDYEGRLIQNSITIERMVLSAFDERRNKYLEEHPSAGKDPKKTGGPIYRRRDGKWVRELILYDKEEIRRIWRQANNGEDATAGLTHLMIWHSNLNDATYQRTRALVRTGMDPRMCSLMQGSTLPRRSGAAGAAVKGVGTMVMELIRMIKRGINDRNFWRGENGRRTRIAYERMCNILKGKFQTAAQRAMMDQVRESRNPGNAEIEDLIFLARSALILRGSVAHKSCLPACVYGLAVASGYDFEREGYSLVGIDPFRLLQNSQVFSLIRPNENPAHKSQLVWMACHSAAFEDLRVSSFIRGTRVAPRGQLSTRGVQIASNENMETMDSSTLELRSRYWAIRTRSGGNTNQQRASAGQISVQPTFSVQRNLPFERATIMAAFTGNTEGRTSDMRTEIIRMMESSRPEDVSFQGRGVFELSDEKATNPIVPSFDMSNEGSYFFGDNAEEYDN",
    "NA":  "MNPNQKIITIGSICMVIGIVSLMLQIGNIISIWVSHSIQTGNQHQAEPCNQSIITYENNTWVNQTYVNISNTNFLTEKAVASVTLAGNSSLCPISGWAVHSKDNGIRIGSKGDVFVIREPFISCSHLECRTFFLTQGALLNDKHSNGTVKDRGPHRTLMSCPVGEAPSPYNSRFESVAWSASACHDGTSWLTIGISGPDNGAVAVLKYNGIITDTIKSWRNNILRTQESECACVNGSCFTVMTDGPSNGQASYKIFKMEKGKVVKSVELNAPNYHYEECSCYPDAGEITCVCRDNWHGSNRPWVSFNQNLEYQIGYICSGVFGDNPRPNDGTGSCGPVSPNGAYGVKGFSFKYGNGVWIGRTKSTNSRSGFEMIWDPNGWTGTDSSFSVKQDIVAITDWSGYSGSFVQHPELTGLDCIRPCFWVELIRGRPKESTIWTSGSSISFCGVNSDTVGWSWPDGAELPFTIDK",
    "M1":   "MSLLTEVETYVLSIVPSGPLKAEIAQRLEDVFAGKNTDLEALMEWLKTRPILSPLTKGILGFVFTLTVPSERGLQRRRFVQNALNGNGDPNNMDRAVKLYKKLKREITFHGAKEVALSYSTGALASCMGLIYNRMGTVTTEVAFGLVCATCEQIADSQHRSHRQMATTTNPLIRHENRMVPASTTAKAMEQMAGSSEQAAEAMEVASQTRQMVQAMRTIGTHPSSSAGLKDNLLENLQAHQKRMGVQMQRFK",
    "NS1":  "MDSNTITSFQVDCYLWHIRKLLSMSDMCDAPFDDRLRRDQKALKGRGSTLGLDLRVATMEGKKIVEDILKSETNENLKIAIASSPAPRYVTDMSIEEMSREWYMLMPRQKITGGLMVKMDQAIMDKRIILKANFSVLFDQLETLVSLRAFTESGAIVAEISPIPSVPGHSTEDVKNAIGILIGGLEWNDNSIRASENIQRFAWGIRDENGGPSLPPKQKRYMAKRVESEV"
}

def get_residue_at_site(segment_name, target_seq_str, site_num):
    """
    Strategy B: Pairwise Alignment
    """
    ref_seq = REF_SEQUENCES.get(segment_name, "")
    target_seq_str = target_seq_str.upper()
    
    try:
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        try:
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        except Exception:
            aligner.match_score = 1.0
            aligner.mismatch_score = -0.5
            
        aligner.open_gap_score = -10.0
        aligner.extend_gap_score = -0.5
        
        alignments = aligner.align(ref_seq, target_seq_str)
        best_alignment = alignments[0]
        
        ref_aligned = best_alignment[0]
        tgt_aligned = best_alignment[1]
        
        current_ref_site = 0
        found_residue = "-"
        
        for r_char, t_char in zip(ref_aligned, tgt_aligned):
            if r_char != '-':
                current_ref_site += 1
            if current_ref_site == site_num:
                found_residue = t_char
                break
        return found_residue

    except Exception as e:
        print(f"Alignment error for {segment_name} site {site_num}: {e}")
        return "-"

# extract_features now returns both Feature Vector and Amino Acid Record
def extract_features(segments_dict: dict):
    """
    Extract amino acids from specified sites in 8 segments and convert to numerical features.
    """
    feature_vector = []
    residue_record = {}
    ordered_segments = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M1", "NS1"]
    
    for seg_name in ordered_segments:
        seq_str = segments_dict.get(seg_name, "")
        sites = SITES_OF_INTEREST.get(seg_name, [])
        for site in sites:
            aa = get_residue_at_site(seg_name, seq_str, site)
            val = SUMMED_FEATURES.get(aa, SUMMED_FEATURES['?'])
            feature_vector.append(val)
            residue_record[f"{seg_name}_{site}"] = aa
            
    return feature_vector, residue_record

def predict_risk(model, scaler, features: list) -> float:
    """
    Use loaded Scaler to transform (standardize) features, then predict.
    """
    features_2d = [features]
    
    try:
        features_scaled = scaler.transform(features_2d)
        risk_probability = model.predict_proba(features_scaled)[0][1]
        return risk_probability
    except ValueError as ve:
        st.error(f"Feature dimension mismatch or standardization error: {ve}.")
        return 0.0
    except Exception as e:
        st.error(f"Prediction process error: {e}")
        return 0.0

def get_example_fasta_content():
    return """>Virus_A|PB2
MERIKELRDLMSQSRTREILTKTTVDHMAIIKKYTSGRQEKNPALRMKWMMAMKYPITADKRITEMVPERNEQGQTLWSK
>Virus_A|PB1
MDVNPTLLFLKVPAQNAISTTFPYTGDPPYSHGTGTGYTMDTVNRTHQYSEKGKWTTNTETGAPQLNPIDGPLPEDNEPS
>Virus_A|PA
MEDFVRQCFNPMIVELAEKAMKEYGEDPKIETNKFAAICTHLEVCFMYSDFHFIDERGESIIVESGDPNALLKHRFEIIE
>Virus_A|HA
MEKIVLLFAIVSLVKSDQICIGYHANNSTEQVDTIMEKNVTVTHAQDILEKKHNGKLCDLDGVKPLILRDCSVAGWLLGN
>Virus_A|NP
MASQGTKRSYEQMETDGERQNATEIRASVGKMIGGIGRFYIQMCTELKLSDYEGRLIQNSLTIERMVLSAFDERRNKYLE
>Virus_A|NA
MNPNQKIITIGSICMVVGIISLILQIGNIISIWVSHSIQTGNQHQNEPISNTNLLTEHGAVCMAWQKQNIAGWNCCTTVT
>Virus_A|M
MSLLTEVETYVLSIVPSGPLKAEIAQRLEDVFAGKNTDLEALMEWLKTRPILSPLTKGILGFVFTLTVPSERGLQRRRFV
>Virus_A|NS
MDSNTVSSFQVDCFLWHVRKRVADQELGDAPFLDRLRRDQKSLRGRGNTLGLDIETATRAGKQIVERILKEESDEALKMT
>Virus_B_Incomplete|HA
MEKIVLLFAIVSLVKSDQICIGYHANNSTEQVDTIMEKNVTVTHAQDILEKKHNGKLCDLDGVKPLILRDCSVAGWLLGN
>Virus_B_Incomplete|NA
MNPNQKIITIGSICMVVGIISLILQIGNIISIWVSHSIQTGNQHQNEPISNTNLLTEHGAVCMAWQKQNIAGWNCCTTVT
"""

def main():
    st.set_page_config(
        page_title="H5 Sentinel",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    with st.container():
        st.title("🦠 H5 Subtype AIV Cross-species Spillover Risk Prediction Tool")
        st.markdown("#### A machine learning-based tool for early warning of H5 Subtype AIV Cross-species Spillover.")
        st.markdown("---")

    with st.expander("ℹ️ Instructions & Input Format Requirements (Click to expand)", expanded=True):
        st.info(
            """
            Please upload a **single** FASTA file containing **Protein/Amino Acid Sequences** for all viruses and all segments.
            
            **FASTA header format must strictly follow `>Virus_Name|Segment_Name` convention.**
            *(e.g., `>A/duck/China/01/2025|HA`)*
            """
        )

    resources = load_resources()
    if resources is None:
        st.stop()
        
    model = resources['model']
    scaler = resources['scaler']

    st.subheader("Step 1: Upload Sequence File")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Don't have data?**")
        st.download_button(
            label="📄 Download Example FASTA",
            data=get_example_fasta_content(),
            file_name="example.fasta",
            mime="text/plain",
            use_container_width=True
        )

    with col2:
        uploaded_file = st.file_uploader(
            "**Upload your FASTA file here** (Supports .fasta, .fas, .fa)", 
            type=["fasta", "fas", "fa"]
        )
    
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        start_button = st.button("🚀 Start Prediction Analysis", type="primary", use_container_width=True)

    if start_button:
        if uploaded_file is not None:
            with st.spinner("🧬 Processing sequences and calculating risk scores... Please wait."):
                complete_viruses, incomplete_viruses, parsing_errors = parse_fasta_file(uploaded_file)
                
                if parsing_errors:
                    st.warning("Issues found during file parsing:")
                    with st.expander("View parsing logs", expanded=False):
                        for error in parsing_errors:
                            st.text(f"- {error}")
            
            if not complete_viruses and not incomplete_viruses:
                st.error("No viruses parsed from file. Please check file content and FASTA header format.")
                return

            results_data = []
            detailed_residues_data = []
            high_risk_count = 0
            
            for name, segments in complete_viruses.items():
                try:
                    features, residue_record = extract_features(segments)
                    risk_score = predict_risk(model, scaler, features)
                    
                    if risk_score > 0.5:
                        high_risk_count += 1

                    results_data.append({
                        "Virus Name": name,
                        "Status": "✅ Processed",
                        "Predicted Spillover Risk (Risk Score)": f"{risk_score:.4f}",
                        "Notes": "-"
                    })
                    
                    residue_record["Virus Name"] = name
                    detailed_residues_data.append(residue_record)
                        
                except Exception as e:
                    results_data.append({
                        "Virus Name": name,
                        "Status": "❌ Prediction Failed",
                        "Predicted Spillover Risk (Risk Score)": "N/A",
                        "Notes": f"Error: {e}"
                    })

            for name, missing in incomplete_viruses:
                results_data.append({
                    "Virus Name": name,
                    "Status": "⚠️ Skipped",
                    "Predicted Spillover Risk (Risk Score)": "N/A",
                    "Notes": f"Missing {len(missing)} segments"
                })

            st.markdown("---")
            st.subheader("Step 2: Analysis Results")

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sequences Analyzed", len(complete_viruses) + len(incomplete_viruses))
            m2.metric("Successfully Processed", len(complete_viruses))
            # Display high risk count if any were processed successfully
            if complete_viruses:
                 m3.metric("High Risk Candidates (>0.5)", high_risk_count, delta=f"{high_risk_count} detected", delta_color="inverse")
            else:
                 m3.metric("Incomplete Sequences", len(incomplete_viruses))

            st.write("#### A. Risk Score Overview")
            if results_data:
                df = pd.DataFrame(results_data)
                def highlight_high_risk(val):
                    try:
                        score = float(val)
                        color = '#ffcccb' if score > 0.5 else '' # Light red for > 0.5
                        return f'background-color: {color}'
                    except:
                        return ''

                df_styled = df.style.apply(
                    lambda x: ['color: red; font-weight: bold;' if "❌" in x or "⚠️" in x else '' for i in x],
                    subset=['Status']
                ).map(highlight_high_risk, subset=['Predicted Spillover Risk (Risk Score)'])
                
                st.dataframe(df_styled, use_container_width=True)
                
                if detailed_residues_data:
                    st.divider()
                    st.write("#### B. Molecular Determinant Details (66 Feature Sites)")
                    st.caption("Download the full amino acid profile using the CSV button on the top right of the table.")
                    
                    df_details = pd.DataFrame(detailed_residues_data)
                    cols = list(df_details.columns)
                    if "Virus Name" in cols:
                        cols.remove("Virus Name")
                        cols = ["Virus Name"] + cols 
                    
                    st.dataframe(df_details[cols], use_container_width=True, height=400)
            else:
                st.warning("No results to display.")

        else:
            st.error("👆 Please upload a FASTA file first to start the analysis.")

if __name__ == "__main__":
    main()