import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

# from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import AgentSelector, wrappers


class BlackjackEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "blackjack_v0"}

    def __init__(self, num_players=1, num_decks=6, render_mode=None):
        super().__init__()
        self.num_players = max(1, min(5, num_players))
        self.num_decks = num_decks
        self.render_mode = render_mode

        # PettingZoo specific attributes
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.possible_agents)
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Game state
        self.deck = None
        self.dealer_hand = []
        self.player_hands = [[] for _ in range(self.num_players)]
        self.player_bets = [0 for _ in range(self.num_players)]
        self.player_double = [1 for _ in range(self.num_players)]

        # Define action and observation spaces for each agent
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "player_hand": spaces.Box(
                        low=0, high=31, shape=(3,), dtype=np.int32
                    ),
                    "dealer_card": spaces.Box(
                        low=0, high=11, shape=(3,), dtype=np.int32
                    ),
                    "visible_cards": spaces.Box(
                        low=np.array([[1, 1] for _ in range(20)]),
                        high=np.array([[13, 4] for _ in range(20)]),
                        shape=(20, 2),
                        dtype=np.int32,
                    ),
                }
            )
            for agent in self.possible_agents
        }

    def render(self):
        if self.render_mode != "human":
            return

        # Print dealer's hand
        dealer_total, dealer_ace = self._calculate_hand(self.dealer_hand)
        dealer_cards = [
            f"{card[0]}{['♠', '♥', '♦', '♣'][card[1] - 1]}" for card in self.dealer_hand
        ]
        print("\nDealer:", " ".join(dealer_cards), f"(Total: {dealer_total})")

        # Print each player's hand
        for i, hand in enumerate(self.player_hands):
            if hand:  # Only show active players
                total, ace = self._calculate_hand(hand)
                cards = [
                    f"{card[0]}{['♠', '♥', '♦', '♣'][card[1] - 1]}" for card in hand
                ]
                bet = f"(Bet: {self.player_bets[i] * self.player_double[i]})"
                status = "BUST!" if total > 21 else ""
                print(f"Player {i}: {' '.join(cards)} {bet} (Total: {total}) {status}")

        print("\n" + "=" * 50 + "\n")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        return self._get_observations()[agent]

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_moves = 0

        # Reset game state
        deck = self._create_deck()
        self.deck = self._riffle_shuffle(deck)
        self.player_bets = [1 for _ in range(self.num_players)]
        self._draw_hands()

        # Initialize agent selector
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        return self._get_observations(), self.infos

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        current_player_idx = int(agent.split("_")[1])

        # Process the action
        match action:
            case 0:  # hit
                self.player_hands[current_player_idx].append(self._draw_card())
                total, _ = self._calculate_hand(self.player_hands[current_player_idx])
                if total > 21:
                    self.terminations[agent] = True

            case 1:  # stand
                self.terminations[agent] = True

            case 2:  # double
                self.player_hands[current_player_idx].append(self._draw_card())
                self.player_double[current_player_idx] = 2
                self.terminations[agent] = True

        if self.terminations[agent]:
            self.agent_selection = self._agent_selector.next()

        # Check if all players are done
        if all(self.terminations.values()):
            # Dealer's turn
            dealer_total, _ = self._calculate_hand(self.dealer_hand)
            while dealer_total < 17:
                self.dealer_hand.append(self._draw_card())
                dealer_total, _ = self._calculate_hand(self.dealer_hand)

            # Calculate rewards
            for idx, agent in enumerate(self.possible_agents):
                player_total, _ = self._calculate_hand(self.player_hands[idx])
                if player_total > 21:
                    self.rewards[agent] = -1
                elif dealer_total > 21 or player_total > dealer_total:
                    self.rewards[agent] = 1
                elif player_total < dealer_total:
                    self.rewards[agent] = -1
                self.rewards[agent] *= self.player_double[idx]

        if self.render_mode == "human":
            self.render()

        return (
            self._get_observations(),
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

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
        return self.deck.pop()

    def _calculate_hand(self, hand):
        # Convert face cards to 10
        values = [min(10, card[0]) for card in hand]
        total = sum(values)
        # Check for ace (rank 1) and if it can be used as 11
        usable_ace = 1 in [card[0] for card in hand] and total + 10 <= 21
        if usable_ace:
            total += 10
        # print(f"hand {hand} value: {total} ace: {usable_ace}")
        return total, usable_ace

    def _draw_hands(self):
        self.player_hands = [[] for _ in range(self.num_players)]
        self.dealer_hand = []
        for _ in range(2):
            for i in range(self.num_players + 1):
                if i == self.num_players:
                    self.dealer_hand.append(self._draw_card())
                elif self.player_bets[i] != 0:
                    self.player_hands[i].append(self._draw_card())

    def _is_round_over(self):
        # Check if all players have terminated (either busted, stood, or doubled)
        return all(self.terminations[agent] for agent in self.agents)

    def _get_observations(self):
        observations = {}
        for idx, agent in enumerate(self.possible_agents):
            total, usable_ace = self._calculate_hand(self.player_hands[idx])
            player_hand = np.array(
                [total, len(self.player_hands[idx]), 1 if usable_ace else 0],
                dtype=np.int32,
            )

            dealer_total, dealer_usable_ace = self._calculate_hand(
                [self.dealer_hand[0]]
            )
            dealer_hand = np.array(
                [dealer_total, 1, 1 if dealer_usable_ace else 0], dtype=np.int32
            )

            visible_cards = self._get_visible_cards()

            observations[agent] = {
                "player_hand": player_hand,
                "dealer_card": dealer_hand,
                "visible_cards": visible_cards,
            }
        return observations

    def _get_visible_cards(self):
        visible_cards = np.zeros((20, 2), dtype=np.int32)
        card_idx = 0

        # Add dealer's visible card
        visible_cards[card_idx] = self.dealer_hand[0]
        card_idx += 1

        # Add all player cards
        for hand in self.player_hands:
            for card in hand:
                if card_idx < 20:
                    visible_cards[card_idx] = card
                    card_idx += 1

        return visible_cards

    # Keep all the existing helper methods (_create_deck, _riffle_shuffle, _shuffle,
    # _draw_card, _calculate_hand, _draw_hands) as they are
    # _draw_card, _calculate_hand, _draw_hands) as they are
