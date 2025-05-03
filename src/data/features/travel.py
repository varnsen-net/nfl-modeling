"""Helper functions for building engineered features for team travel."""


import polars as pl


def get_city_coordinates(teams, loc_replacements):
    """Returns a dataframe with the decimal latitude and longitude for every
    NFL team city in the Sharpe teams data.

    Mostly saving this just for reference. I only need to run it once.

    :param pd.DataFrame teams: Team names and locations
    :param dict loc_replacements: Location replacements
    :return: City names, latitudes, and longitudes
    :rtype: pd.DataFrame
    """
    cities = teams['location'].unique()
    cities = pd.Series(cities, name='location')
    cities_adj = cities.replace(loc_replacements)
    lat = []
    lon = []
    for city in cities_adj:
        print(city)
        url = f'https://nominatim.openstreetmap.org/search?q={city}&format=json'
        response = requests.get(url).json()
        lat.append(response[0]['lat'])
        lon.append(response[0]['lon'])
    city_coords = pd.DataFrame({'city':cities, 'lat':lat, 'lon':lon})
    city_coords = pd.concat([cities, city_coords[['lat', 'lon']]], axis=1)
    return city_coords


def attach_lats_lons(games, city_coords):
    """Merge the city coordinates onto the games dataframe.
    
    :param pl.LazyFrame games: Raw games dataframe.
    :param pl.DataFrame city_coords: City coordinates dataframe.
    :return: Games dataframe with city coordinates.
    :rtype: pl.LazyFrame
    """
    city_coords = city_coords.select('name', 'lat', 'lon')
    return (
        games
        .join(city_coords, left_on='away_team', right_on='name', how='left')
        .rename({'lat': 'lat_away', 'lon': 'lon_away'})
        .join(city_coords, left_on='home_team', right_on='name', how='left')
        .rename({'lat': 'lat_home', 'lon': 'lon_home'})
    )


def add_travel_distances(games):
    """Calculate the haversine distance traveled by the away team for each game.
    
    :param pl.LazyFrame games: Games dataframe with lat/lon columns.
    :return: Games dataframe with travel distances.
    :rtype: pl.LazyFrame
    """
    return (
        games
        .with_columns(
            lat_away_rad=(3.14159 / 180) * pl.col('lat_away'),
            lon_away_rad=(3.14159 / 180) * pl.col('lon_away'),
            lat_home_rad=(3.14159 / 180) * pl.col('lat_home'),
            lon_home_rad=(3.14159 / 180) * pl.col('lon_home'),
        )
        .with_columns(
            dlat=pl.col('lat_home_rad') - pl.col('lat_away_rad'),
            dlon=pl.col('lon_home_rad') - pl.col('lon_away_rad'),
        )
        .with_columns(
            a=((pl.col('dlat') / 2).sin().pow(2)
                + pl.col('lat_away_rad').cos()
                * pl.col('lat_home_rad').cos()
                * (pl.col('dlon') / 2).sin().pow(2))
        )
        .with_columns(
            c=2 * pl.arctan2(pl.col('a').sqrt(), (1 - pl.col('a')).sqrt())
        )
        .with_columns(
            away_travel_distance=6373 * pl.col('c'),  # km
            home_travel_distance=0,
        )
    )


def add_coord_deltas(games):
    """Calculate the difference in longitude and latitude between the home and
    away teams.

    :param pl.LazyFrame games: Games dataframe with lat/lon columns.
    :return: Games dataframe with coordinate deltas.
    :rtype: pl.LazyFrame
    """
    return (
        games
        .with_columns(
            away_lon_delta=pl.col('lon_home') - pl.col('lon_away'),
            away_lat_delta=pl.col('lat_home') - pl.col('lat_away'),
            home_lon_delta=0,
            home_lat_delta=0,
        )
    )


def round_travel_values(games):
    """Round the travel distances and coordinate deltas to 0 or 1 decimal.

    :param pl.LazyFrame games: Games dataframe with travel distances and deltas.
    :return: Games dataframe with rounded values.
    :rtype: pl.LazyFrame
    """
    return (
        games
        .with_columns(
            away_travel_distance=pl.col('away_travel_distance').round(0),
            home_travel_distance=pl.col('home_travel_distance').round(0),
            away_lon_delta=pl.col('away_lon_delta').round(1),
            away_lat_delta=pl.col('away_lat_delta').round(1),
            home_lon_delta=pl.col('home_lon_delta').round(1),
            home_lat_delta=pl.col('home_lat_delta').round(1),
        )
    )


def build_travel_features(raw_games, city_coords):
    """Build engineered features for team travel.

    :param pl.LazyFrame raw_games: Raw games dataframe.
    :param pl.DataFrame city_coords: City coordinates dataframe.
    :return: Games dataframe with travel features.
    :rtype: pl.LazyFrame
    """
    games = attach_lats_lons(raw_games, city_coords)
    games = add_travel_distances(games)
    games = add_coord_deltas(games)
    games = round_travel_values(games)
    return (
        games
        .select(
            'game_id',
            'away_travel_distance',
            'home_travel_distance',
            'away_lon_delta',
            'away_lat_delta',
            'home_lon_delta',
            'home_lat_delta',
        )
        .sort('game_id')
    )
