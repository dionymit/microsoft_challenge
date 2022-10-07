# AIYungsters present to you: Contoso spots
We developed a smart vest that guides you during your routes!<br>
Our innovative integrated LEDs show the way for routes recommend to you. It tracks your routes and you can challenge people and celebreties. Those are recommend using our AI services based on duration and distance of similar routes.

<img src="presentation/recommendation.png" style="width:300px"><br>
# Team
<img src="presentation/team.jpeg" style="width:300px"><br>
Left to right: <br>
Maximilian Studt<br>
Dionysis Mitosios<br>
Azur Causevic<br>
Konrad Wehner <br>
Thore Koritzius<br>

# Images
Deployement worked locally, however, we have issues with the ML container remote

<img src="presentation/start.png" style="width:300px"><br>
<img src="presentation/tracks.png" style="width:300px"><br>
<img src="presentation/choose.png" style="width:300px"><br>
<img src="presentation/recommend.png" style="width:300px"><br>

# Software Stack
<img src="presentation/stack.png" style="width:500px"><br>
The Backend utilitzes a python flask server connected to our PostrgreSQL.<br>
There we deployed three databases: routes, tracks, accounts.<br>
Our Model is exported as .pkl from the Azure ML container. We then do inference on it in our backend.<br>
To simulate data from the user, he can select avg distance and duration and get a match<br>