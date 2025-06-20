
## File Structure

### Tool Folder

- `microTry.py` - Contains the prompt for MicroLens
- `movieTry.py` - Contains the prompt for MovieLens
- `crawler.py` - Web crawler for scraping movie information from MovieLens
- `youtube.py` - Crawler that maps MovieLens IDs to YouTube IDs and downloads trailers
- `google.py` - Supplementary crawler for trailers not found via YouTube.py (searches Google for missing trailers)

### Data File

- `final.csv` - Collected movie dataset containing the following fields:

## Dataset Fields Description

| Column Name | Description |
|-------------|-------------|
| id | MovieLens unique identifier for the movie |
| describe | Brief summary/description of the movie's content |
| language | Primary language of the movie |
| directors | List of directors for the movie |
| cast | Main cast members/actors in the movie |
| poster | URL or path to the movie poster image |
| genre | Movie genre(s) or category(ies) |
| tag | User-generated tags from MovieLens |
| youtube | YouTube ID or link for the movie's trailer |

## Workflow Process

1. **Data Collection**:
   - `crawler.py` scrapes basic movie info from MovieLens
   
2. **Trailer Acquisition**:
   - Primary method: `youtube.py` finds trailers using MovieLens-Youtube ID mapping
   - Secondary method: `google.py` searches for trailers not found via primary method
   - Final method: Manual download for any remaining missing trailers

3. **Data Compilation**:
   - All collected data is combined into `final.csv`

## Usage

To use the crawlers, run:
```bash
python crawler.py
python youtube.py
python google.py