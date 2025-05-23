(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		hook - tool
		yogurt - box
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(beyondworkspace rack)
		(on rack table)
		(on hook table)
		(on salt rack)
		(on yogurt rack)
		(inhand icecream)
	)
	(:goal (and))
)
