This readme.txt file was generated on 2020-09-28 by Anders Dohlman at Duke Univ.


GENERAL INFORMATION

1. Title of Dataset

	"The Cancer Microbiome Atlas (TCMA): A Pan-Cancer Comparative Analysis to Distinguish Organ-Associated Microbiota from Equiprevalent Contaminants"


2. Author Information
	
	A. Principal Investigator Contact Information
		Name: Xiling Shen
		Institution: Duke University
		Address: Durham, NC
		Email: xiling.shen@duke.edu


3. Dates of data collection: 

	2017-09-01 to 2020-09-01.


4. Keywords

	Human microbiome, The Cancer Genome Atlas, Colorectal cancer, Contamination, Host-microbe interactions, Multi-omics, Pan-cancer


5. Information about funding sources that supported the collection of the data:  

	This study was supported by NIH R35GM122465, DK119795 and DARPA W911NF1920111.


DATA & FILE OVERVIEW


Fields reference:

	assay		data were generated from whole-genome (WGS) or whole-exome (WGS) sequencing
	tissue-type	data included represents solid tissue samples or blood samples
	data-level	data were aggregated at the sample, case, or file (ie. sequencing run) level
	normalization	centered log ratio (clr), relative abundance (relabund), or reads


1. File List: 

Taxonomy dictionary (tsv)

	taxonomy.txt

Metadata files (tsv) are of the format:

	"metadata.<assay>.<tissue_type>.<data-level>.txt"

	metadata.WGS.blood.case.txt
	metadata.WGS.blood.file.txt
	metadata.WGS.blood.sample.txt
	metadata.WGS.solid.case.txt
	metadata.WGS.solid.file.txt
	metadata.WGS.solid.sample.txt
	metadata.WXS.blood.case.txt
	metadata.WXS.blood.file.txt
	metadata.WXS.blood.sample.txt
	metadata.WXS.solid.case.txt
	metadata.WXS.solid.file.txt
	metadata.WXS.solid.sample.txt

Data files (tsv) are of the format:

	"bacteria.<assay>.<tissue-type>.<data-level>.<normalization>.txt"

	bacteria.WGS.blood.case.relabund.txt
	bacteria.WGS.blood.file.reads.txt
	bacteria.WGS.blood.sample.clr.txt
	bacteria.WGS.blood.sample.relabund.txt
	bacteria.WGS.solid.case.clr.txt
	bacteria.WGS.solid.case.relabund.txt
	bacteria.WGS.solid.file.reads.txt
	bacteria.WGS.solid.sample.clr.txt
	bacteria.WGS.solid.sample.relabund.txt
	bacteria.WXS.blood.case.clr.txt
	bacteria.WXS.blood.case.relabund.txt
	bacteria.WXS.blood.file.reads.txt
	bacteria.WXS.blood.sample.clr.txt
	bacteria.WXS.blood.sample.relabund.txt
	bacteria.WXS.solid.case.clr.txt
	bacteria.WXS.solid.case.relabund.txt
	bacteria.WXS.solid.file.reads.txt
	bacteria.WXS.solid.sample.clr.txt
	bacteria.WXS.solid.sample.relabund.txt

PhyloSeq objects (Rds) are of the format:

	"physeq.<assay>.<tissue-type>.<data-level>.<normalization>.rds"

	physeq.WGS.blood.case.clr.rds
	physeq.WGS.blood.case.relabund.rds
	physeq.WGS.blood.file.reads.rds
	physeq.WGS.blood.sample.clr.rds
	physeq.WGS.blood.sample.relabund.rds
	physeq.WGS.solid.case.clr.rds
	physeq.WGS.solid.case.relabund.rds
	physeq.WGS.solid.file.reads.rds
	physeq.WGS.solid.sample.clr.rds
	physeq.WGS.solid.sample.relabund.rds
	physeq.WXS.solid.case.clr.rds
	physeq.WXS.solid.case.relabund.rds
	physeq.WXS.solid.file.reads.rds
	physeq.WXS.solid.sample.clr.rds
	physeq.WXS.solid.sample.relabund.rds

2. File format

	All text files are tab-separated format (tsv). The Rsv (R single-object storage) contains a phyloseq object which can be opened in R and is read using the phyloseq package.


3. Data relationship

	Data columns and metadata rows are matched by assay/tissue-type/data-level. Data rows contain NCBI taxonomy ids defined in taxonomy.txt.


4. Date the files were created

	2020-09-29


5. Date the files were updated

	N/A


6. Relation to external datasets

	Data sample and case barcodes match the TCGA naming conventions. For file-level data, the TCGA experiment UUID is given. All metadata were obtained through the GDC API. Metadata variable descriptions can be found at the GDC documentation (https://docs.gdc.cancer.gov/API/).


SHARING AND ACCESS INFORMATION

1. Licenses

	The data is available via Creative Commons Zero (CC0). Users are free to use the database and its content in new and different ways, provided they provide attribution to the source of the data and/or the database.


2. Links to publications

	Dohlman et al. Cell Host & Microbe (2020). In review.


3. Links to publicly accessible locations

	Explore the database at http://tcma.pratt.duke.edu


4. Recommended citation for the data

	Dohlman, Anders B.; Arguijo Mendoza,  Diana; Ding, Shengli; Gao, Michael; Dressman, Holly; Iliev, Iliyan D.; Lipkin, Steven M.; Shen, Xiling (In review). "The Cancer Microbiome Atlas (TCMA): A Pan-Cancer Comparative Analysis to Distinguish Organ-Associated Microbiota from Equiprevalent Contaminants". Cell Host & Microbe. 


METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
	
	Studying the microbial composition of internal organs and their associations with disease remains challenging due to the difficulty of acquiring clinical biopsies. We designed a statistical model to analyze the prevalence of species across sample types from The Cancer Genome Atlas (TCGA), revealing that species equiprevalent across sample types are predominantly contaminants, bearing unique signatures from each TCGA-designated sequencing center. Removing such species mitigated batch effects and isolated the tissue-resident microbiome, which was validated with original matched TCGA samples. “Mixed-evidence” species can be further distinguished by gene copy and nucleotide variants. We thus present The Cancer Microbiome Atlas (TCMA), a collection of curated, decontaminated microbial compositions of oropharyngeal, esophageal, gastrointestinal, and colorectal tissues. This led to discovery of prognostic species and blood signatures of mucosal barrier injuries, and enabled systematic matched microbe-host multi-omics analyses, which will help guide future studies of the microbiome’s role in human health and disease.


2. Information of data processing

	After isolating the tissue-resident population, file-level were aggregated to the sample and case level by averaging and renormalizing relative abundances. The CLR transform maps data to a continuous, unbounded domain that is normally distributed. The centered log ratio transform was calculated over taxa with greater than 25% prevalence. More information can be found in the Methods section of the associated manuscript.





