def search_movie(query: str, movies) -> list[str]:
    results: list[str] = []
    for movie in movies:
        if query.lower() in movie["title"].lower():
            results.append(movie)
    return results
