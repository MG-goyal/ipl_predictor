from predict import predict_match_winner

if __name__ == "__main__":
    sample = {
        'team1': 'Mumbai Indians',
        'team2': 'Chennai Super Kings',
        'toss_winner': 'Mumbai Indians',
        'toss_decision': 1,
        'venue': 'Wankhede Stadium',
        'predicted_score': 170,
        'defendable': 1
    }

    print(predict_match_winner(sample))