import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
STRATEGY_PATH = SCRIPT_DIR / "strategy.csv"


class BlackjackEnv(gym.Env):
    def __init__(
        self, max_num_players: int = 5, hands_per_check: int = 20, num_decks: int = 6
    ):
        self.max_num_players = max_num_players
        self.num_players = max_num_players
        self.hands_per_check = hands_per_check
        self.played_hands = 0
        self.active_player = 0
        self.active_player_played_hands = [{"total": 0, "bet": 0}]
        self.deck = []
        self.bet = 0
        self.num_decks = num_decks
        self.current_cards = []
        self.last_cards = []
        self.dealer_hand = []
        self.player_hands = [[] for _ in range(self.num_players)]
        self.strategy_df = pd.read_csv(STRATEGY_PATH)
        self.can_double = True

        self.observation_space = gym.spaces.Box(
            low=np.array([[0, 0] for _ in range(100)], dtype=np.float32),
            high=np.array([[13, 4] for _ in range(100)], dtype=np.float32),
            shape=(100, 2),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(11)  # 0 to 10 tokens

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.num_players = np.random.randint(1, self.max_num_players)
        self.active_player = np.random.randint(0, self.num_players)
        self.current_cards = []
        self.last_cards = []
        self.played_hands = 0

        deck = self._create_deck()
        self.deck = self._riffle_shuffle(deck)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        bet = action
        self._draw_hands(bet)
        for player in range(self.num_players):
            is_active = self.active_player == player
            self._play_hand(self.player_hands[player], bet.copy(), is_active)

        dealer_total, _ = self._calculate_hand(self.dealer_hand)
        while dealer_total < 17:
            self.dealer_hand.append(self._draw_card())
            dealer_total, _ = self._calculate_hand(self.dealer_hand)

        self.last_cards = self.current_cards
        self.current_cards = []

        reward = self._calculate_reward()
        obs = self._get_obs()

        # Shuffle decks
        self._shuffle()

        self.played_hands += 1
        done = False

        if self.played_hands >= self.hands_per_check:
            done = True
            self.played_hands = 0

        return obs, reward, done, False, {}

    def _draw_hands(self, bet):
        self.player_hands = [[] for _ in range(self.num_players)]
        self.dealer_hand = []
        for _ in range(2):
            for i in range(self.num_players + 1):
                if i == self.num_players:
                    self.dealer_hand.append(self._draw_card())
                elif i == self.active_player and bet != 0:
                    self.player_hands[i].append(self._draw_card())
                else:
                    self.player_hands[i].append(self._draw_card())

        # print("player_hands", self.player_hands)
        # print("dealer hand", self.dealer_hand)

    def _get_obs(self):
        obs = np.zeros((100, 2), dtype=np.float32)
        for i, card in enumerate(self.last_cards[:100]):
            obs[i] = [card[0], card[1]]
        return obs

    def _play_hand(self, hand: list, bet: int, is_active):
        stop = False
        split_hands = []
        while not stop:
            play_action = self._strategy(hand, self.dealer_hand)
            match play_action:
                case "H":
                    hand.append(self._draw_card())
                case "S":
                    stop = True
                case "D":
                    if self.can_double:
                        if is_active:
                            bet *= 2
                        stop = True
                    hand.append(self._draw_card())
                case "E":
                    if self.can_double:
                        if is_active:
                            bet *= 2
                        hand.append(self._draw_card())
                    stop = True
                case "P":
                    split_hands = self._split(hand, bet.copy(), is_active)
                    stop = True

            hand_total, _ = self._calculate_hand(hand)
            if hand_total >= 21:
                stop = True
        if not split_hands:
            self.active_player_played_hands.append({"total": hand_total, "bet": bet})
        else:
            for split_hand in split_hands:
                if (
                    isinstance(split_hand, list)
                    and len(split_hand) > 0
                    and isinstance(split_hand[0], list)
                ):
                    # Handle nested split hands (when a split hand was split again)
                    for nested_hand in split_hand:
                        hand_total, _ = self._calculate_hand(nested_hand)
                        self.active_player_played_hands.append(
                            {"total": hand_total, "bet": bet}
                        )
                else:
                    # Handle regular split hand
                    hand_total, _ = self._calculate_hand(split_hand)
                    self.active_player_played_hands.append(
                        {"total": hand_total, "bet": bet}
                    )
        return hand

    def _split(self, hand, bet, is_active):
        handA = [hand[0]]
        handB = [hand[1]]

        handA.append(self._draw_card())
        final_handA = self._play_hand(handA, bet.copy(), is_active)
        handB.append(self._draw_card())
        final_handB = self._play_hand(handB, bet.copy(), is_active)
        return [final_handA, final_handB]

    def _strategy(self, hand, dealer_hand) -> str:
        hand_total, _ = self._calculate_hand(hand)
        if hand_total == 21:
            return "S"

        # Parse hand value to strategy format
        values = [min(10, card[0]) for card in hand]
        if len(values) == 2:
            if 1 in values:
                other_values = [v for v in values if v != 1]
                if not other_values:
                    other_value = "A"
                else:
                    other_value = other_values[0]
                hand_cards = f"A{other_value}"
            elif values[0] == values[1]:
                hand_cards = f"{values[0]}{values[1]}"
            else:
                hand_cards, _ = self._calculate_hand(hand)
                hand_cards = str(hand_cards)
        else:
            hand_cards, _ = self._calculate_hand(hand)
            hand_cards = str(hand_cards)

        # Parse dealer card to strategy format
        if dealer_hand[0][0] == 1:
            dealer_card = "A"
        else:
            dealer_card = str(min(10, dealer_hand[0][0]))
        play_action = self.strategy_df.loc[
            self.strategy_df["Hand"] == hand_cards, dealer_card
        ].iloc[0]

        return play_action

    def _calculate_reward(self):
        total_reward = 0

        for hand_info in self.active_player_played_hands:
            hand_total = hand_info["total"]
            hand_bet = hand_info["bet"]
            dealer_total, _ = self._calculate_hand(self.dealer_hand)

            if hand_total > 21:
                total_reward -= hand_bet
            elif dealer_total > 21:
                total_reward += hand_bet
            elif hand_total > dealer_total:
                total_reward += hand_bet
            elif hand_total < dealer_total:
                total_reward -= hand_bet
            # tie case: no reward change
        self.active_player_played_hands = [
            {"total": 0, "bet": 0}
        ]  # Reset for next round
        return total_reward

    def _create_deck(self):
        suits = [1, 2, 3, 4]  # 1=Spades, 2=Hearts, 3=Diamonds, 4=Clubs
        ranks = list(range(1, 14))  # 1=Ace, 11=Jack, 12=Queen, 13=King
        deck = []
        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    deck.append((rank, suit))
        return deck

    def _riffle_shuffle(self, deck, riffles=7):
        deck = deck.copy()  # Create a copy to avoid modifying the original
        if len(deck) <= 1:
            return deck

        for _ in range(riffles):  # Casino standard is 7 riffles
            split_point = len(deck) // 2 + np.random.randint(
                -5, 6
            )  # Simulate imperfect splits
            left = deck[:split_point]
            right = deck[split_point:]

            # Riffle shuffle simulation
            deck = []
            while left or right:
                # Randomly choose which pile to draw from, with slight bias variations
                if left and (
                    not right or np.random.random() < 0.48
                ):  # Slight bias to one side
                    deck.append(left.pop(0))
                elif right:
                    deck.append(right.pop(0))

        # Final cut
        cut_point = np.random.randint(0, len(deck))
        deck = deck[cut_point:] + deck[:cut_point]
        return deck

    def _shuffle(self):
        # Extract next 21 cards (if available)
        next_cards = self.deck[:21] if len(self.deck) >= 21 else self.deck[:]
        remaining_cards = self.deck[21:] if len(self.deck) >= 21 else []
        shuffled_reamining_cards = self._riffle_shuffle(remaining_cards)
        # Put next_cards back on top
        self.deck = next_cards + shuffled_reamining_cards

    def _draw_card(self):
        if not self.deck:
            self.deck = self._create_deck()
        card = self.deck.pop()
        self.current_cards.append(card)
        return card

    def _calculate_hand(self, hand):
        # Convert face cards to 10
        values = [min(10, card[0]) for card in hand]
        total = sum(values)
        # Check for ace (rank 1) and if it can be used as 11
        usable_ace = 1 in [card[0] for card in hand] and total + 10 <= 21
        if usable_ace and total + 10 <= 21:
            total += 10
        return total, usable_ace


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    # This will catch many common issues
    try:
        env = BlackjackEnv()
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
