# AIYungsters present to you: Contoso spots
We developed a smart west that guides you during your routes!<br>
Our innovative integrated LEDs show the way for routes recommend to you. It tracks your routes and you can challenge people and celebreties. Those are recommend using our AI services based on duration and distance of similar routes.

# Team
<img src="presentation/team.jpeg" style="width:300px"><br>
Left to right: <br>
Maximilian Studt<br>
Dionysis Mitosios<br>
Azur Causevic<br>
Konrad Wehner <br>
Thore Koritzius<br>

# Software Stack
The Backend utilitzes a python flask server connected to our PostrgreSQL.<br>
There we deployed three databases: routes, tracks, accounts.<br>
Our Model is exported as .pkl from the Azure ML container. We then do inference on it in our backend.<br>
To simulate data from the user, he can select avg distance and duration and get a match<br>