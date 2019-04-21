The final processed 3' UTR multi-library APA dataset.

File structure and content:

step3_data_isoforms.zip : Compressed archive of the combined, processed 3' UTR library data, with isoform measurements. In this version of the library, the cleavage position counts are aggregated and stored as proximal/non-proximal counts.
	- apa_general3_antimisprime_orig.csv : Combined library csv-file containing Variant sequence-APA measurement entries
		Columns:
		seq : 186bp variant UTR sequence. pPAS located at position 50.
		seq_ext: 256bp variant UTR sequence. pPAS located at position 125 (padded with more wildtype sequence).
		proximal_count : # unique UMI reads mapping to the proximal region of cleavage.
		total_count_vs_all : Total # of unique UMI reads mapping anywhere on the sequence.
		total_count_vs_distal: Total # of unique UMI reads mapping to either the proximal or distal region of cleavage.
		proximal_avgcut : Average cleavage position within the proximal region of cleavage.
		proximal_stdcut : Standard deviation in cleavage position within the proximal region of cleavage.
		library : Unique library identifier
		library_name: Library name
Processed feature data from the csv-file (stored as numpy nd-arrays):
	- apa_general3_antimisprime_orig_avgcut.npy : Average cleavage position within the proximal region of cleavage.
	- apa_general3_antimisprime_orig_count.npy : Total # of unique UMI reads mapping anywhere on the sequence.
	- apa_general3_antimisprime_orig_distalpas.npy : 1 if there is a fixed, non-random, dPAS downstream of the variant sequence.
	- apa_general3_antimisprime_orig_ispas.npy : 1 if any read supporting proximal cleavage, 0 otherwise
	- apa_general3_antimisprime_orig_libindex.npy : Unique library identifier
	- apa_general3_antimisprime_orig_output.npy : Column 0 : proximal_count / total_count_vs_distal. Column 1 : proximal_count / total_count_vs_all. Column 2 : (total_count_vs_all - total_count_vs_distal + proximal_count) / total_count_vs_all.
	- apa_general3_antimisprime_orig_stdcut.npy : Standard deviation in cleavage position within the proximal region of cleavage.
	- npz_apa_seq_general3_antimisprime_orig_input.npz : One-hot-encoded matrix of variant sequences.
	- npz_apa_seq_general3_antimisprime_orig_pasaligned_input.npz : One-hot-encoded matrix of variant sequences. Sequences were aligned so that the strongest pPAS is at position 50.
	- npz_apa_seq_general3_antimisprime_orig_features.npz : Column 0 : Seq contains Canonical pPAS AATAAA. Column 1 : Seq contains Canonical pPAS ATTAAA. Column 2 : Seq contains competing PAS AATAAA. Column 3 : Seq contains competing PAS ATTAAA. Column 4 : Seq contains 1-base mut of competing PAS AATAAA.


step3_data_cuts.zip : Compressed archive of the combined, processed 3' UTR library data, with raw cleavage position measurements. In this version of the library, the cleavage position counts are stored without any aggregation.
	- apa_general_cuts_antimisprime_orig.csv : Combined library csv-file containing Variant sequence-APA measurement entries
		Columns:
		seq : 186bp variant UTR sequence. pPAS located at position 50.
		seq_ext: 256bp variant UTR sequence. pPAS located at position 125 (padded with more wildtype sequence).
		total_count : Total # of unique UMI reads mapping anywhere on the sequence.
		library : Unique library identifier
		library_name: Library name
	- apa_general_cuts_antimisprime_orig_cutdistribution.mat : Matlab dictionary file containing a sparse scipy csr_matrix (under key 'cuts') of cleavage position probabilities. Rows specify unique variants and columns specify cut position. Probabilities were calculated as # of reads mapping to a position / Total # of reads.

Processed feature data from the csv-file (stored as numpy nd-arrays):
	- apa_general3_antimisprime_orig_count.npy : Total # of unique UMI reads mapping anywhere on the sequence.
	- apa_general3_antimisprime_orig_distalpas.npy : 1 if there is a fixed, non-random, dPAS downstream of the variant sequence.
	- apa_general3_antimisprime_orig_libindex.npy : Unique library identifier
	- npz_apa_seq_general3_antimisprime_orig_input.npz : One-hot-encoded matrix of variant sequences.
	- npz_apa_seq_general3_antimisprime_orig_pasaligned_input.npz : One-hot-encoded matrix of variant sequences. Sequences were aligned so that the strongest pPAS is at position 50.
	- npz_apa_seq_general3_antimisprime_orig_features.npz : Column 0 : Seq contains Canonical pPAS AATAAA. Column 1 : Seq contains Canonical pPAS ATTAAA. Column 2 : Seq contains competing PAS AATAAA. Column 3 : Seq contains competing PAS ATTAAA. Column 4 : Seq contains 1-base mut of competing PAS AATAAA.
