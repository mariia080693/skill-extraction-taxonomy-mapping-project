import json
import csv
from collections import Counter
import matplotlib.pyplot as plt

from src.config import JOBS_PATH, TAXONOMY_PATH

# Set a dark gray background for all plots
plt.rcParams['figure.facecolor'] = '#404040'  # 25% lighter than black
plt.rcParams['axes.facecolor'] = '#404040'
plt.rcParams['savefig.facecolor'] = '#404040'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['figure.edgecolor'] = '#404040'


print("\nJOBS EXPLORATION:\n")


# --- Load data ---
jobs = []
with open(JOBS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        jobs.append(json.loads(line))

total_jobs = len(jobs)
print("Total number of jobs:", total_jobs, "\n")


# --- Duplicates ---
id_counts = Counter(j["id"] for j in jobs)
dups = sum(1 for v in id_counts.values() if v > 1)
print("Duplicated rows number:", dups, "\n")

title_counts = Counter(j["title"] for j in jobs)
print('Unique job titles:', len(title_counts), "\n")
print("Top 20 repeated titles:")
for title, count in title_counts.most_common(20):
    print(title, "---", count)
print("\n")

# Bar chart of top 5 most repeated job titles
top5 = title_counts.most_common(5)
labels = [t for t, _ in top5] # extract just the titles for labeling
values = [c for _, c in top5] 

plt.figure(figsize=(7, 5))
plt.bar(labels, values, color='lightgreen')
plt.ylabel("Count")
plt.title("Top 5 Most Repeated Job Titles")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()



# --- One sample job listing ---
sample = jobs[0]
print("Sample job listing:")
print("Top-level keys:", list(sample.keys()))
print("Metadata keys:", list(sample['metadata'].keys()), "\n")

print("\nSample jobs entry:")
print(jobs[0],"\n")

# --- Field completeness ---
def pct(n):
    return f"{n:,} ({n/total_jobs*100:.0f}%)"

fields = {
    "title": sum(1 for j in jobs if j.get("title")),
    "abstract": sum(1 for j in jobs if j.get("abstract")),
    "content": sum(1 for j in jobs if j.get("content")),
    "classification": sum(1 for j in jobs if j["metadata"].get("classification", {}).get("name")),
    "subClassification": sum(1 for j in jobs if j["metadata"].get("subClassification", {}).get("name")),
    "location": sum(1 for j in jobs if j["metadata"].get("location", {}).get("name")),
    "workType": sum(1 for j in jobs if j["metadata"].get("workType", {}).get("name")),
    "salaryText": sum(1 for j in jobs if j["metadata"].get("additionalSalaryText")),
}

print("Field completeness:")
for name, count in fields.items():
    print(name,"---", pct(count))
print("\n")


# --- Unique classifications count ---
classifications = Counter(
    j["metadata"]["classification"]["name"]
    for j in jobs if j["metadata"].get("classification", {}).get("name")
)
print("Unique classifications:", len(classifications))
for name, count in classifications.most_common(15):
    print(name, "---", pct(count))
print("\n")

# Bar chart of top 5 most repeated classifications
top5 = classifications.most_common(5)
labels = [t for t, _ in top5] # extract just the titles for labeling
values = [c for _, c in top5] 

plt.figure(figsize=(7, 5))
plt.bar(labels, values, color='lightblue')
plt.ylabel("Count")
plt.title("Top 5 Most Repeated Classifications")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# --- Unique sub-classifications ---
sub_cls = Counter(
    j["metadata"]["subClassification"]["name"]
    for j in jobs if j["metadata"].get("subClassification", {}).get("name")
)
print("Unique sub-classifications:", len(sub_cls), "\n")


# --- Unique locations ---
locations = Counter(
    j["metadata"]["location"]["name"]
    for j in jobs if j["metadata"].get("location", {}).get("name")
)
print("Unique locations:", len(locations), "\n")


# --- Taxonomy exploration ---
print("\nTAXONOMY EXPLORATION:\n")

taxonomy = []
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        taxonomy.append(row)
        
print("\nSample taxonomy entry:")
print(taxonomy[0], "\n")

occupations = set(dict["occupationLabel"] for dict in taxonomy)
skills = set(dict["skillLabel"] for dict in taxonomy)
print("Total number of paired occupation-skills:", len(taxonomy)) 
print("Occupations:", len(occupations))
print("Skills:", len(skills))
print("Relation types:", dict(Counter(dict['relationType'] for dict in taxonomy)))
print("Skill types:   ", dict(Counter(dict['skillType'] for dict in taxonomy)))

# Bar chart of top 5 occupations by number of skills
occ_skill_counts = Counter(row["occupationLabel"] for row in taxonomy)
top5_occ = occ_skill_counts.most_common(5)
occ_labels = [o for o, _ in top5_occ]
occ_values = [c for _, c in top5_occ]

plt.figure(figsize=(7, 5))
plt.bar(occ_labels, occ_values, color='lightcoral')
plt.ylabel("Count")
plt.title("Top 5 Occupations by Number of Skills")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()


