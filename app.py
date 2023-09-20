import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn


st.title('Air Quality Index Prediction')
st.image('images.png')

#Load data
preprocessor = pd.read_pickle('Preprocessor.pkl')
model = pd.read_pickle('Model.pkl')

# App
Category = st.selectbox('Category',['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Hazardous', 'Very Unhealthy'])
Defining_Parameter = st.selectbox('Defining Parameter',['PM2.5', 'Ozone', 'NO2', 'PM10', 'CO'])
Number_of_Sites_Reporting = st.number_input('Number of Sites Reporting', min_value=1, max_value=72)
city_ascii = st.selectbox('City',['Aberdeen', 'Akron', 'Albany', 'Albuquerque', 'Amarillo','Americus', 'Ann Arbor', 'Appleton', 'Ardmore', 'Arkadelphia','Ashtabula', 'Athens', 'Atlanta', 'Augusta', 'Austin','Bakersfield', 'Baltimore', 'Bangor', 'Baraboo', 'Baton Rouge',
'Beatrice', 'Beaumont', 'Beaver Dam', 'Bellingham', 'Bend','Billings', 'Bishop', 'Bismarck', 'Blacksburg', 'Bloomington','Boston', 'Boulder', 'Bowling Green', 'Bozeman', 'Bremerton',
 'Brownsville', 'Brunswick', 'Buffalo', 'Butte', 'Cadillac','Cambridge', 'Canton', 'Cape Coral', 'Carlsbad', 'Cedar City','Cedar Rapids', 'Centralia', 'Champaign', 'Charleston',
 'Charlotte', 'Charlottesville', 'Chattanooga', 'Chicago', 'Chico','Cincinnati', 'Claremont', 'Clarksville', 'Cleveland', 'Clinton','College Station', 'Colorado Springs', 'Columbia', 'Columbus',
 'Corning', 'Corpus Christi', 'Corsicana', 'Corvallis', 'Crestview','Dallas', 'Dalton', 'Danville', 'Daphne', 'Davenport', 'Dayton','Decatur', 'Deming', 'Denver', 'Des Moines', 'Dickinson',
'Douglas', 'Duluth', 'Durango', 'Eagle Pass', 'Eau Claire','Effingham', 'El Centro', 'El Paso', 'Elizabethtown', 'Ellensburg','Eugene', 'Evansville', 'Fargo', 'Farmington', 'Fayetteville',
'Flagstaff', 'Florence', 'Fort Collins', 'Fort Madison','Fort Payne', 'Fort Smith', 'Fort Wayne', 'Fresno', 'Gadsden','Gainesville', 'Gettysburg', 'Gillette', 'Glenwood Springs',
'Grand Island', 'Grand Junction', 'Grants Pass', 'Great Falls','Greeley', 'Green Bay', 'Greenville', 'Guayama', 'Gulfport','Hagerstown', 'Hanford', 'Harrison', 'Harrisonburg', 'Hattiesburg','Helena', 'Hermiston', 'Hilo', 'Hobbs', 'Homosassa Springs','Houma', 'Houston', 
'Huntington', 'Idaho Falls', 'Indianapolis','Iowa City', 'Ithaca', 'Jackson', 'Jacksonville', 'Jamestown','Jasper', 'Jefferson City', 'Joplin', 'Kalispell', 'Kansas City',
'Kennewick', 'Killeen', 'Kingsport', 'Kingsville', 'Knoxville','La Crosse', 'La Grande', 'Lafayette', 'Lake Charles', 'Lake City','Lake Havasu City', 'Lakeland', 'Laramie', 'Laredo', 'Las Cruces',
'Las Vegas', 'Lawton', 'Lewiston', 'Lexington', 'Lima', 'Lincoln','Little Rock', 'Logan', 'Longview', 'Lubbock', 'Lynchburg','Macon', 'Madison', 'Marietta', 'Marshall', 'Mason City',
 'Mayaguez', 'McAlester', 'McAllen', 'Medford', 'Memphis', 'Merced','Meridian', 'Michigan City', 'Milwaukee', 'Minot', 'Missoula','Mobile', 'Modesto', 'Monroe', 'Montgomery', 'Morehead City',
 'Morgantown', 'Moses Lake', 'Mount Vernon', 'Muncie', 'Muscatine','Myrtle Beach', 'Naples', 'Nashville', 'New Castle', 'New Orleans','New York', 'Nogales', 'Ocala', 'Odessa', 'Ogden', 'Oklahoma City',
'Olympia', 'Omaha', 'Orlando', 'Othello', 'Oxnard', 'Palatka','Panama City', 'Parkersburg', 'Payson', 'Pensacola', 'Peoria','Philadelphia', 'Phoenix', 'Pittsburgh', 'Pittsfield','Platteville', 'Pocatello', 'Ponca City', 'Ponce', 'Port Angeles',
 'Portland', 'Portsmouth', 'Prescott', 'Price', 'Providence','Provo', 'Pueblo', 'Pullman', 'Quincy', 'Raleigh', 'Red Bluff','Redding', 'Reno', 'Richmond', 'Riverside', 'Riverton',
'Roanoke Rapids', 'Roanoke', 'Rochester', 'Rock Springs','Rockford', 'Rockland', 'Rome', 'Sacramento', 'Salem','Salt Lake City', 'San Antonio', 'San Diego', 'San Francisco',
'San Jose', 'San Juan', 'Santa Fe', 'Santa Maria', 'Santa Rosa','Savannah', 'Scottsbluff', 'Seattle', 'Sebastian', 'Sebring','Sevierville', 'Shelton', 'Sheridan', 'Show Low', 'Shreveport',
'Sierra Vista', 'Sioux City', 'Somerset', 'Sonora', 'South Bend''Spartanburg', 'Spokane', 'Springfield', 'St. George','St. Joseph', 'St. Louis', 'St. Marys', 'State College',
 'Steamboat Springs', 'Stockton', 'Summerville', 'Syracuse','Tampa', 'Taos', 'Texarkana', 'Toledo', 'Trenton', 'Truckee','Tucson', 'Tulsa', 'Tupelo', 'Tuscaloosa', 'Tyler', 'Utica',
 'Valdosta', 'Vallejo', 'Vernal', 'Victoria', 'Vincennes','Vineyard Haven', 'Virginia Beach', 'Visalia', 'Wabash', 'Waco','Walla Walla', 'Warner Robins', 'Washington Court House',
'Washington', 'Waterloo', 'Watertown', 'Weirton', 'Wenatchee','Wheeling', 'Wilmington', 'Winchester', 'Worcester', 'Yakima','Youngstown', 'Yuba City', 'Yuma', 'Adjuntas', 'Adrian',
'Alexandria', 'Allentown', 'Altoona', 'Anchorage', 'Asheville','Atlantic City', 'Bartlesville', 'Bay City', 'Bemidji', 'Berlin',  'Birmingham', 'Brainerd', 'Branson', 'Bridgeport', 'Brookings','Burlington', 'Carson City', 'Casper', 'Chambersburg', 'Cheyenne','Clarksburg', 'Clearlake', "Coeur d'Alene", 'Concord','Cookeville', 'Crescent City', 'Cullowhee', 'Deltona', 'Detroit','Dodge City', 'Dover', 'DuBois', 'Durham', 'Dyersburg','East Stroudsburg', 'El Dorado', 'Elkhart', 'Elko', 'Emporia','Erie', 'Eureka', 'Evanston', 'Fairbanks', 'Fairmont', 'Fallon','Fernley', 'Flint', 'Fond du Lac', 'Gardnerville Ranchos','Grand Rapids', 'Greensboro', 'Hailey', 'Hammond', 'Harrisburg','Hartford', 'Hickory', 'Holland', 'Hot Springs', 'Houghton','Huntsville', 'Indiana', 'Janesville', 'Johnstown', 'Juneau','Kahului', 'Kalamazoo', 'Kapaa', 'Keene', 'Kinston', 'Klamath Falls', 'Kokomo', 'Laconia', 'Lancaster', 'Lansing','Lawrenceburg', 'Lebanon', 'Los Alamos', 'Los Angeles','Ludington', 'Madera', 'Manchester', 'Manitowoc', 'Miami','Middlesborough', 'Minneapolis', 'Morristown', 'Moscow','Muskegon', 'Napa', 'New Haven', 'Niles', 'North Port', 'Norwich', 'Owensboro', 'Oxford', 'Paducah', 'Pahrump', 'Palm Bay', 'Pierre','Port St. Lucie', 'Prineville', 'Racine', 'Rapid City', 'Reading','Red Wing', 'Rocky Mount', 'Roseburg', 'Roswell', 'Ruidoso','Rutland', 'Salinas', 'Salisbury', 'San Luis Obispo', 'Sandpoint','Santa Cruz', 'Sayre', 'Scranton', 'Sheboygan', 'Sioux Falls','St. Cloud', 'Tallahassee', 'Terre Haute', 'The Dalles', 'Topeka','Torrington', 'Traverse City', 'Twin Falls', 'Ukiah', 'Vineland','Wausau', 'Whitewater', 'Wichita', 'Williamsport', 'Winona','York', 'Coos Bay', 'Dothan', 'Georgetown', 'Jonesboro','Muskogee', 'Sault Ste. Marie', 'Seneca', 'Seymour', 'Walterboro','Weatherford', 'Williston', 'Craig', 'Durant', 'Grenada','Montrose', 'Sanford', 'Greenwood', 'Tahlequah', 'Talladega','Ames', 'Gaffney', 'Beckley', 'Boone', 'Crawfordsville','Goldsboro', 'Marion', 'Lumberton', 'Silver City', 'Wichita Falls','Gallup', 'Manhattan', 'Shawnee', 'Breckenridge', 'Elmira','Kingston', 'Laurel', 'Russellville', 'Burley', 'Findlay','Frankfort', 'Fredericksburg', 'Natchez', 'Searcy', 'Bucyrus','Carbondale', 'Iron Mountain', 'Ottawa', 'Coffeyville','Fort Morgan', 'Ogdensburg', 'Sterling', 'Alamogordo', 'Coco','Hilton Head Island', 'Pottsville', 'Safford', 'Eufaula','Lawrence', 'London', 'Troy', 'Blackfoot', 'Bluefield','Blytheville', 'DeRidder', 'Enid', 'Morgan City', 'Mountain Home','North Platte', 'Picayune', 'Saginaw', 'Scottsboro', 'Tullahoma','Vicksburg', 'Aguadilla', 'Elizabeth City', 'Mayfield','Orangeburg', 'Point Pleasant', 'Tiffin', 'Union City','Binghamton', 'Hudson', 'Oshkosh', 'Pine Bluff', 'Stillwater','Warsaw', 'Willmar', 'Wisconsin Rapids', 'Alpena', 'Auburn','Barre', 'Central City', 'Corinth', 'Albert Lea', 'Battle Creek','Edwards', 'Fremont', 'Grand Forks', 'Greeneville', 'Ketchikan','Key West', 'Mankato', 'Port Clinton', 'Susanville', 'West Plains','Big Stone Gap', 'Dunn', 'Fergus Falls', 'Hutchinson', 'Mansfield','Bellefontaine', 'Clarksdale', 'Rexburg', 'Sandusky', 'Anniston',
'Cumberland', 'Johnson City', 'Kearney', 'Marquette','Martinsville', 'Mexico', 'Paris', 'Bedford', 'Camden','Garden City', 'Wilson', 'Woodward', 'Big Rapids', 'Glens Falls','Grants', 'Hannibal', 'Hastings', 'Ruston', 'Alma', 'Midland','New Philadelphia', 'Staunton', 'Escanaba', 'Lewisburg', 'Pontiac',
'Norwalk', 'Dubuque', 'Cullman', 'Kankakee', 'Maysville','Newport', 'Glasgow', 'Madisonville', 'Murray', 'New Bern','Ocean City', 'Celina', 'Borger', 'Menomonie'])
state_name = st.selectbox('state name',['Washington', 'Ohio', 'Georgia', 'Oregon', 'New York','New Mexico', 'Texas', 'Michigan', 'Wisconsin', 'Oklahoma','Arkansas', 'Maine', 'California', 'Maryland', 'Louisiana','Nebraska', 'Montana', 'North Dakota', 'Virginia', 'Illinois',
'Indiana', 'Massachusetts', 'Colorado', 'Kentucky', 'Florida','Utah', 'Iowa', 'West Virginia', 'South Carolina','North Carolina', 'Tennessee', 'New Hampshire', 'Mississippi', 'Missouri', 'Alabama', 'Minnesota', 'Arizona', 'Pennsylvania','Wyoming', 'Puerto Rico', 'Hawaii', 'Idaho', 'Nevada',
'Rhode Island', 'New Jersey', 'District of Columbia','South Dakota', 'Alaska', 'Connecticut', 'Vermont', 'Kansas','Delaware'])
timezone = st.selectbox('timezone',['America/Los_Angeles', 'America/New_York', 'America/Denver','America/Chicago', 'America/Detroit','America/Indiana/Indianapolis', 'America/Matamoros','America/Phoenix', 'America/Puerto_Rico', 'Pacific/Honolulu',
'America/Boise', 'America/Indiana/Vincennes', 'America/Anchorage','America/Juneau', 'America/Menominee', 'America/Toronto','America/Sitka'])
Year = st.number_input('Year of measurement',min_value=1980, max_value=2022)
Months = st.number_input('Month of measurement',min_value=1, max_value=12 )
Day = st.number_input('Day of measurement',min_value=1, max_value=30 )



new_data = {'Category': Category , 'Defining Parameter': Defining_Parameter  ,
            'Number of Sites Reporting':Number_of_Sites_Reporting, 'city_ascii':city_ascii ,
            'state_name':state_name , 'timezone':timezone  , 'Year':Year , 
           'Month':Months,'Day': Day}

new_data = pd.DataFrame(new_data,index=[0])

#Preprocessed
new_data_preprocessed = preprocessor.transform(new_data)

AQI = model.predict(new_data_preprocessed)

# Output
if st.button('Predict'):
    st.markdown('## Air Quality Index:')
    st.markdown(AQI)
