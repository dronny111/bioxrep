from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PublicSource:
    key: str
    url: str
    filename: str
    description: str
    track: str
    compressed: bool = False
    credentialed: bool = False


PUBLIC_SOURCES: Dict[str, PublicSource] = {
    "clinvar_variant_summary": PublicSource(
        key="clinvar_variant_summary",
        url="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
        filename="variant_summary.txt.gz",
        description="ClinVar variant summary with genes, variation IDs, clinical significance, locations, and names.",
        track="variant",
        compressed=True,
    ),
    "clinvar_hgvs": PublicSource(
        key="clinvar_hgvs",
        url="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/hgvs4variation.txt.gz",
        filename="hgvs4variation.txt.gz",
        description="ClinVar HGVS strings keyed by variation ID; useful for cross-notation equivalence classes.",
        track="variant",
        compressed=True,
    ),
    "clinvar_allele_gene": PublicSource(
        key="clinvar_allele_gene",
        url="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/allele_gene.txt.gz",
        filename="allele_gene.txt.gz",
        description="ClinVar allele-to-gene mappings.",
        track="variant",
        compressed=True,
    ),
    "hgnc_complete_set": PublicSource(
        key="hgnc_complete_set",
        url="https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
        filename="hgnc_complete_set.txt",
        description="HGNC approved gene symbols, aliases, previous symbols, RefSeq, Ensembl, UniProt, and MANE IDs.",
        track="gene_alias",
    ),
    "hgnc_withdrawn": PublicSource(
        key="hgnc_withdrawn",
        url="https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/withdrawn.txt",
        filename="hgnc_withdrawn.txt",
        description="HGNC withdrawn and merged/split symbol reports.",
        track="gene_alias",
    ),
    "mimiciv_hosp_d_labitems": PublicSource(
        key="mimiciv_hosp_d_labitems",
        url="https://physionet.org/files/mimiciv/3.1/hosp/d_labitems.csv.gz",
        filename="d_labitems.csv.gz",
        description="MIMIC-IV v3.1 hospital lab item dictionary; requires credentialed PhysioNet access.",
        track="clinical_labs",
        compressed=True,
        credentialed=True,
    ),
    "mimiciv_hosp_labevents": PublicSource(
        key="mimiciv_hosp_labevents",
        url="https://physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz",
        filename="labevents.csv.gz",
        description="MIMIC-IV v3.1 hospital laboratory events; requires credentialed PhysioNet access.",
        track="clinical_labs",
        compressed=True,
        credentialed=True,
    ),
    "mimiciv_hosp_patients": PublicSource(
        key="mimiciv_hosp_patients",
        url="https://physionet.org/files/mimiciv/3.1/hosp/patients.csv.gz",
        filename="patients.csv.gz",
        description="MIMIC-IV v3.1 patient demographics table; requires credentialed PhysioNet access.",
        track="clinical_labs",
        compressed=True,
        credentialed=True,
    ),
    "mimiciv_hosp_admissions": PublicSource(
        key="mimiciv_hosp_admissions",
        url="https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz",
        filename="admissions.csv.gz",
        description="MIMIC-IV v3.1 hospital admissions table; requires credentialed PhysioNet access.",
        track="clinical_labs",
        compressed=True,
        credentialed=True,
    ),
}
